"""
train.py
========
Training loop for AdaptiveFusionModel.
- Differential learning rates (CLIP vs fusion layers)
- Linear warmup scheduler
- Early stopping
- Checkpointing best model by AUROC
"""

import os
import torch
import torch.nn as nn
import numpy as np
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from dataset import get_dataloaders
from model import AdaptiveFusionModel


# ─────────────────────────────────────────────────────────────────
# CONFIG — change these to match your setup
# ─────────────────────────────────────────────────────────────────
CONFIG = {
    "train_parquet" : "../Data/train-00000-of-00001-6587b3a58d350036.parquet",
    "val_parquet"   : "../Data/validation-00000-of-00001-1508d9e5032c2c1f.parquet",  # ← your filename
    "batch_size"    : 16,
    "num_epochs"    : 10,
    "freeze_clip"   : True,    # True = faster, less memory. False = fine-tune everything
    "clip_lr"       : 1e-5,    # small LR for pretrained CLIP
    "fusion_lr"     : 1e-4,    # larger LR for your new layers
    "weight_decay"  : 0.01,
    "warmup_ratio"  : 0.1,     # 10% of steps used for warmup
    "pos_weight"    : 1.5,     # class imbalance correction (hateful = minority)
    "checkpoint_dir": "checkpoints",
    "patience"      : 3        # early stopping patience
}


# ─────────────────────────────────────────────────────────────────
# TRAIN ONE EPOCH
# ─────────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_idx, batch in enumerate(loader):
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values   = batch['pixel_values'].to(device)
        labels         = batch['label'].to(device)

        optimizer.zero_grad()

        logits, alpha = model(input_ids, attention_mask, pixel_values)
        loss = criterion(logits, labels)

        loss.backward()

        # Gradient clipping — prevents exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.extend(probs)
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

    auroc = roc_auc_score(all_labels, all_preds)
    return total_loss / len(loader), auroc


# ─────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_alphas = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values   = batch['pixel_values'].to(device)
            labels         = batch['label'].to(device)

            logits, alpha = model(input_ids, attention_mask, pixel_values)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_alphas.extend(alpha.mean(dim=-1).cpu().numpy())  # mean alpha per sample

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_alphas = np.array(all_alphas)

    bin_preds = (all_preds >= 0.5).astype(int)

    auroc    = roc_auc_score(all_labels, all_preds)
    accuracy = accuracy_score(all_labels, bin_preds)
    f1       = f1_score(all_labels, bin_preds, average='macro')

    return total_loss / len(loader), auroc, accuracy, f1, all_alphas


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    # FOR M2 MAC
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    # ── Data ─────────────────────────────────────────────────────
    train_loader, val_loader = get_dataloaders(
        CONFIG["train_parquet"],
        CONFIG["val_parquet"],
        batch_size=CONFIG["batch_size"]
    )

    # ── Model ─────────────────────────────────────────────────────
    model = AdaptiveFusionModel(freeze_clip=CONFIG["freeze_clip"]).to(device)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # ── Loss (with class imbalance correction) ────────────────────
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([CONFIG["pos_weight"]]).to(device)
    )

    # ── Optimizer (differential LRs) ──────────────────────────────
    optimizer = torch.optim.AdamW([
        {
            'params': model.clip.parameters(),
            'lr': CONFIG["clip_lr"]          # small LR for CLIP
        },
        {
            'params': model.cross_attn.parameters(),
            'lr': CONFIG["fusion_lr"]        # larger LR for fusion layers
        },
        {
            'params': model.gating.parameters(),
            'lr': CONFIG["fusion_lr"]
        },
        {
            'params': model.classifier.parameters(),
            'lr': CONFIG["fusion_lr"]
        },
    ], weight_decay=CONFIG["weight_decay"])

    # ── Scheduler (linear warmup) ─────────────────────────────────
    total_steps   = CONFIG["num_epochs"] * len(train_loader)
    warmup_steps  = int(total_steps * CONFIG["warmup_ratio"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # ── Training Loop ─────────────────────────────────────────────
    best_auroc   = 0.0
    patience_ctr = 0

    for epoch in range(CONFIG["num_epochs"]):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch+1}/{CONFIG['num_epochs']}")
        print(f"{'='*50}")

        train_loss, train_auroc = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_auroc, val_acc, val_f1, alphas = evaluate(
            model, val_loader, criterion, device
        )

        print(f"\nTrain → Loss: {train_loss:.4f} | AUROC: {train_auroc:.4f}")
        print(f"Val   → Loss: {val_loss:.4f} | AUROC: {val_auroc:.4f} | "
              f"Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
        print(f"Alpha → Mean: {alphas.mean():.3f} | "
              f"Std: {alphas.std():.3f} | "
              f"Min: {alphas.min():.3f} | Max: {alphas.max():.3f}")

        # ── Save best model ───────────────────────────────────────
        if val_auroc > best_auroc:
            best_auroc   = val_auroc
            patience_ctr = 0
            path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pt")
            torch.save({
                'epoch'     : epoch + 1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_auroc' : val_auroc,
                'val_acc'   : val_acc,
            }, path)
            print(f"✅ Saved best model → AUROC: {best_auroc:.4f}")
        else:
            patience_ctr += 1
            print(f"No improvement. Patience: {patience_ctr}/{CONFIG['patience']}")
            if patience_ctr >= CONFIG["patience"]:
                print("Early stopping triggered.")
                break

    print(f"\nTraining complete. Best Val AUROC: {best_auroc:.4f}")


if __name__ == "__main__":
    main()
