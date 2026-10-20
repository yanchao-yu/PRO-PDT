import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from recognition.config import (
    DEVICE,
    NUM_EPOCHS,
    VIT_WEIGHTS,
    EVALUATION_DIR,
    BATCH_SIZE,
    PRETRAINED_VIT_WEIGHTS,
    VIT_CUSTOM_PARAMS
)
from recognition.core.utils.model_loader import ModelLoader, ModelRegistry
from recognition.core.utils.plot_curves import plot_curves
from recognition.core.utils.dataset import TRAIN_DATASET, VAL_DATASET

print(f"Using DEVICE: {DEVICE.type}")

train_loader = DataLoader(
    TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=8, pin_memory=True, persistent_workers=True
)
print(f"Training on {len(TRAIN_DATASET.samples)} identities")
val_loader = DataLoader(
    VAL_DATASET, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=8, pin_memory=True
)
print(f"Validating on {len(VAL_DATASET.samples)} identities")

num_classes = len(set(label for _, label in TRAIN_DATASET.samples))
print(f"Training on {num_classes} identities")

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
    missing = [k for k in model_state if k not in matched]
    unexpected = [k for k in checkpoint if k not in model_state]
    model.load_state_dict(matched, strict=False)
    print(f"Matched   : {len(matched)} / {len(model_state)} layers")
    print(f"Missing   : {len(missing)} layers — {missing[:3]}")
    print(f"Unexpected: {len(unexpected)} keys")
else:
    print("WARNING: No pretrained weights found, training from random init")

# 3. ArcFace loss with reduced margin for fine-tuning
arcface_loss = ModelLoader.load(ModelRegistry.ArcFace, overrides={
    "in_features": 512,
    "out_features": num_classes,
    "m": VIT_CUSTOM_PARAMS['arcface_m'],  # reduced from 0.5 — easier convergence for fine-tuning
    "s": VIT_CUSTOM_PARAMS['arcface_s']  # reduced from 64.0 — reduces loss scale
}).to(DEVICE)

# 4. Discriminative optimizer
optimizer = torch.optim.AdamW([
    {"params": model.parameters(), "lr": VIT_CUSTOM_PARAMS['lr_backbone'], "weight_decay": VIT_CUSTOM_PARAMS['weight_decay']},
    {"params": arcface_loss.parameters(), "lr": VIT_CUSTOM_PARAMS['lr_arcface'], "weight_decay": VIT_CUSTOM_PARAMS['weight_decay']}
])

# 5. Label smoothing to reduce overfitting
criterion = nn.CrossEntropyLoss(label_smoothing=VIT_CUSTOM_PARAMS['label_smooth'])

# 6. Mixed precision scaler
scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# 7. Scheduler with warmup
warmup_epochs = VIT_CUSTOM_PARAMS['warmup_epochs']
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        ),
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=NUM_EPOCHS - warmup_epochs, eta_min=1e-6
        )
    ],
    milestones=[warmup_epochs]
)

# 8. Early stopping
patience = 10
no_improve_count = 0

best_val_acc = 0.0
train_losses, train_accs, val_accs = [], [], []

print(f"Starting training for {NUM_EPOCHS} epochs...")

for epoch in range(NUM_EPOCHS):
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

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        no_improve_count = 0
        torch.save({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "arcface": arcface_loss.state_dict(),
            "val_acc": val_acc
        }, VIT_WEIGHTS)
        print(f"The best weights has been saved on epoch : {epoch + 1}")
    else:
        no_improve_count += 1

    print(
        f"Epoch {epoch + 1:02d}/{NUM_EPOCHS:02d} | "
        f"Loss: {epoch_train_loss:.4f} | "
        f"Train Acc: {epoch_train_acc:.1f}% | "
        f"Val Acc: {val_acc:.1f}% | "
        f"LR backbone: {optimizer.param_groups[0]['lr']:.6f} | "
        f"LR arcface: {optimizer.param_groups[1]['lr']:.6f}"
    )

    if no_improve_count >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}. "
              f"Best val acc: {best_val_acc:.1f}%")
        break

plot_curves(
    train_losses=train_losses,
    train_accs=train_accs,
    val_accs=val_accs,
    model_name=f"ViT_Face_Transformer_{NUM_EPOCHS}epochs",
    save_dir=os.path.join(EVALUATION_DIR, "vision_transformer/plots")
)
