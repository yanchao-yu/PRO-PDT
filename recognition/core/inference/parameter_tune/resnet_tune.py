import json
import os
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1

from recognition.config import DEVICE, BATCH_SIZE, EVALUATION_DIR
from recognition.core.utils.plot_curves import plot_curves
from recognition.core.utils.dataset import TRAIN_DATASET, VAL_DATASET
from recognition.core.utils.model_loader import ModelLoader, ModelRegistry

# Fixed path name to match actual model
TUNING_DIR = os.path.join(EVALUATION_DIR, "inception_resnet_v1", "hyperparameter_tuning")
CHECKPOINT_DIR = os.path.join(TUNING_DIR, "checkpoints")
PLOT_DIR = os.path.join(TUNING_DIR, "plots")
os.makedirs(TUNING_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

train_loader = DataLoader(
    TRAIN_DATASET,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

val_loader = DataLoader(
    VAL_DATASET, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=8, pin_memory=True
)

print(f"Validating Images:  {len(VAL_DATASET.samples)}")
print(f"Training Images: {len(TRAIN_DATASET.samples)}")

num_classes = len(set(label for _, label in TRAIN_DATASET.samples))
print(f"Training on {num_classes} identities")

SEARCH_EPOCHS = 50


def objective(trial):
    # ---------------------------------------------------------
    # 1. Hyperparameters
    # ---------------------------------------------------------
    lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 5e-5, log=True)
    lr_arcface = trial.suggest_float("lr_arcface", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    label_smooth = trial.suggest_float("label_smoothing", 0.0, 0.2)
    arcface_m = trial.suggest_float("arcface_m", 0.2, 0.5)
    arcface_s = trial.suggest_float("arcface_s", 16.0, 64.0)
    warmup_epochs = trial.suggest_int("warmup_epochs", 2, 5)

    print(f"\n{'=' * 70}\nTRIAL {trial.number} | Params: {trial.params}\n{'=' * 70}")
    print(f"Using DEVICE: {DEVICE.type}")

    # ---------------------------------------------------------
    # 2. Model & Layer Unfreezing (BEFORE OPTIMIZER!)
    # ---------------------------------------------------------
    model = InceptionResnetV1(pretrained="vggface2", classify=False).to(DEVICE)

    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze specific layers
    layers_to_unfreeze = [
        "block8",
        "avgpool_1a",
        "dropout",
        "last_linear",
        "last_bn",
    ]

    for name, module in model.named_children():
        if name in layers_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True

    # ---------------------------------------------------------
    # 3. ArcFace, Optimizer & Scheduler Setup
    # ---------------------------------------------------------
    arcface_loss = ModelLoader.load(ModelRegistry.ArcFace, overrides={
        "in_features": 512,
        "out_features": num_classes,
        "m": arcface_m,
        "s": arcface_s
    }).to(DEVICE)

    # CRITICAL FIX: Only pass parameters that require grad
    trainable_backbone_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": trainable_backbone_params, "lr": lr_backbone, "weight_decay": weight_decay},
        {"params": arcface_loss.parameters(), "lr": lr_arcface, "weight_decay": weight_decay}
    ])

    criterion = nn.CrossEntropyLoss(label_smoothing=label_smooth)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

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

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    patience = 10
    no_improve_count = 0
    best_val_acc = 0.0
    train_losses, train_accs, val_accs = [], [], []

    for epoch in range(SEARCH_EPOCHS):
        model.train()
        arcface_loss.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                embeddings = model(images)
                logits = arcface_loss(embeddings, labels)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                trainable_backbone_params + list(arcface_loss.parameters()),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        scheduler.step()

        acc = (correct / total) * 100
        epoch_loss = total_loss / len(train_loader)
        train_losses.append(epoch_loss)
        train_accs.append(acc)

        # ---------------------------------------------------------
        # 5. Validation Loop (Pure Cosine Similarity)
        # ---------------------------------------------------------
        model.eval()
        arcface_loss.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            normalized_class_weights = F.normalize(arcface_loss.weight, p=2, dim=1)

            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                    embeddings = model(images)
                    normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
                    val_logits = torch.matmul(normalized_embeddings, normalized_class_weights.T)

                val_correct += (val_logits.argmax(dim=1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = (val_correct / val_total) * 100
        val_accs.append(val_acc)
        trial.report(val_acc, epoch)

        # Optuna Pruning
        if trial.should_prune():
            del model, arcface_loss, optimizer, scaler
            torch.cuda.empty_cache()
            raise optuna.TrialPruned()

        # Checkpointing
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
            f"Loss: {epoch_loss:.4f} | "
            f"Train: {acc:.1f}% | "
            f"Val: {val_acc:.1f}% | "
            f"Best: {best_val_acc:.1f}%"
        )

        if no_improve_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. Best val acc: {best_val_acc:.1f}%")
            break

    plot_curves(
        train_losses=train_losses,
        train_accs=train_accs,
        val_accs=val_accs,
        model_name=f"InceptionResnetV1_{trial.number}",
        save_dir=PLOT_DIR
    )

    # Full GPU Memory Cleanup
    del model, arcface_loss, optimizer, scaler
    torch.cuda.empty_cache()

    return best_val_acc


# ---------------------------------------------------------
# 6. Study Execution
# ---------------------------------------------------------
study = optuna.create_study(
    study_name="inception_resnet_v1_tuning",
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=5)
)
study.optimize(objective, n_trials=20)

# Save full results table
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