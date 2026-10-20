import json
import os
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from recognition.config import (DEVICE, EVALUATION_DIR,
                                BATCH_SIZE, PRETRAINED_VIT_WEIGHTS)
from recognition.core.utils.model_loader import ModelLoader, ModelRegistry
from recognition.core.utils.plot_curves import plot_curves
from recognition.core.utils.dataset import TRAIN_DATASET, VAL_DATASET

TUNING_DIR = os.path.join(EVALUATION_DIR, "vision_transformer", "hyperparameter_tuning")
CHECKPOINT_DIR = os.path.join(TUNING_DIR, "checkpoints")
PLOT_DIR = os.path.join(TUNING_DIR, "plots")
os.makedirs(TUNING_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

train_loader = DataLoader(
    TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=8, pin_memory=True, persistent_workers=True
)
val_loader = DataLoader(
    VAL_DATASET, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=8, pin_memory=True
)
print(f"Validating Images:  {len(VAL_DATASET.samples)}")
print(f"Training Images: {len(TRAIN_DATASET.samples)}")

num_classes = len(set(label for _, label in TRAIN_DATASET.samples))
print(f"Training on {num_classes} identities")

SEARCH_EPOCHS = 30


def objective(trial):
    # --------------------------------
    # 1. Hyperparameters to tune
    # --------------------------------
    lr_backbone = trial.suggest_float("lr_backbone", 1e-5, 1e-3, log=True)
    lr_arcface = trial.suggest_float("lr_arcface", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-4, 5e-4, log=True)
    arcface_m = trial.suggest_float("arcface_m", 0.2, 0.5)
    arcface_s = trial.suggest_float("arcface_s", 16.0, 64.0)
    label_smooth = trial.suggest_float("label_smoothing", 0.0, 0.2)
    warmup_epochs = trial.suggest_int("warmup_epochs", 2, 5)

    print(f"\n{'=' * 70}")
    print(f"TRIAL {trial.number} | Params: {trial.params}")
    print(f"{'=' * 70}")
    print(f"Using DEVICE: {DEVICE.type}")

    # 1. Load model backbone
    model = ModelLoader.load(
        ModelRegistry.ViTFace,
        overrides={"num_class": num_classes}
    ).to(DEVICE)

    # 2. Load pretrained Face-Transformer weights
    if os.path.exists(PRETRAINED_VIT_WEIGHTS):
        print(f"Loading pretrained weights from: {PRETRAINED_VIT_WEIGHTS}")
        checkpoint = torch.load(PRETRAINED_VIT_WEIGHTS, map_location=DEVICE)
        model_state = model.state_dict()
        matched = {
            k: v for k, v in checkpoint.items()
            if k in model_state and v.shape == model_state[k].shape
        }
        model.load_state_dict(matched, strict=False)
        print(f"Matched {len(matched)}/{len(model_state)} layers")

    else:
        print("WARNING: No pretrained weights found, training from random init")

    # 3. ArcFace loss with reduced margin for fine-tuning
    arcface_loss = ModelLoader.load(ModelRegistry.ArcFace, overrides={
        "in_features": 512,
        "out_features": num_classes,
        "m": arcface_m,
        "s": arcface_s
    }).to(DEVICE)

    # 4. Discriminative optimizer
    optimizer = torch.optim.AdamW([
        {"params": model.parameters(), "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": arcface_loss.parameters(), "lr": lr_arcface, "weight_decay": weight_decay}
    ])

    # 5. Label smoothing to reduce overfitting
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)

    # 6. Mixed precision scaler
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    # 7. Scheduler with warmup
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_epochs
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=SEARCH_EPOCHS - warmup_epochs, eta_min=1e-6
            )
        ],
        milestones=[warmup_epochs]
    )

    # 8. Early stopping
    patience = 10
    no_improve_count = 0
    best_val_acc = 0.0
    train_losses, train_accs, val_accs = [], [], []

    print(f"Starting training for {SEARCH_EPOCHS} epochs...")

    for epoch in range(SEARCH_EPOCHS):
        model.train()
        arcface_loss.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=DEVICE.type,
                                    enabled=(DEVICE.type == "cuda")):
                embeddings = model(images)
                logits = arcface_loss(embeddings, labels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(arcface_loss.parameters()),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        epoch_train_loss = total_loss / len(train_loader)
        epoch_train_acc = correct / total * 100
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # validation
        model.eval()
        arcface_loss.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with torch.amp.autocast(device_type=DEVICE.type,
                                        enabled=(DEVICE.type == "cuda")):
                    embeddings = model(images)
                    # cosine similarity for validation — no label needed
                    val_logits = F.linear(
                        F.normalize(embeddings, p=2, dim=1),
                        F.normalize(arcface_loss.weight, p=2, dim=1)
                    ) * arcface_loss.s

                val_correct += (val_logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total * 100
        val_accs.append(val_acc)
        trial.report(val_acc, epoch)

        if trial.should_prune():
            del model, arcface_loss, optimizer, scaler
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0

            torch.save({
                "trial": trial.number,
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "arcface": arcface_loss.state_dict(),
                "val_acc": val_acc,
                "hyperparameters": trial.params
            }, os.path.join(CHECKPOINT_DIR, f"trial_{trial.number}_best.pth"))
        else:
            no_improve_count += 1

        print(
            f"Epoch {epoch + 1:02d}/{SEARCH_EPOCHS} | "
            f"Loss: {epoch_train_loss:.4f} | "
            f"Train: {epoch_train_acc:.1f}% | "
            f"Val: {val_acc:.1f}% | "
            f"Best: {best_val_acc:.1f}%"
        )

        if no_improve_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. "
                  f"Best val acc: {best_val_acc:.1f}%")
            break

    plot_curves(
        train_losses=train_losses,
        train_accs=train_accs,
        val_accs=val_accs,
        model_name=f"VIT_TRIAL_{trial.number}",
        save_dir=PLOT_DIR
    )

    # cleanup GPU memory between trials
    del model, arcface_loss, optimizer, scaler
    torch.cuda.empty_cache()

    return best_val_acc


study = optuna.create_study(study_name="vit_tuning", direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5))
study.optimize(objective, n_trials=20)

# save full results table
trials_df = study.trials_dataframe()
trials_df.to_csv(os.path.join(TUNING_DIR, "all_trials.csv"), index=False)

with open(os.path.join(TUNING_DIR, "best_parameters.json"), "w") as f:
    json.dump({
        "best_params": study.best_params,
        "best_val_acc": study.best_value,
        "best_trial_number": study.best_trial.number,
        "n_trials": len(study.trials)
    }, f, indent=4)

print("\nBest trial:")
print(f"  Number : {study.best_trial.number}")
print(f"  Val Acc: {study.best_value:.2f}%")
print(f"  Params :")
for k, v in study.best_params.items():
    print(f"    {k}: {v}")
