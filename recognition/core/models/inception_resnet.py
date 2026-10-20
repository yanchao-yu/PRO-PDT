import os

import torch
import torch.nn as nn
from facenet_pytorch import InceptionResnetV1
from torch.utils.data import DataLoader
import torch.nn.functional as F

from recognition.config import DEVICE, BATCH_SIZE, EVALUATION_DIR, NUM_EPOCHS, RESNET_WEIGHTS, RESNET_PARAMS
from recognition.core.utils.plot_curves import plot_curves
from recognition.core.utils.dataset import TRAIN_DATASET, VAL_DATASET
from recognition.core.utils.model_loader import ModelLoader, ModelRegistry

print(f"DEVICE: {DEVICE}")


def train_resnet():
    train_loader = DataLoader(
        TRAIN_DATASET,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )



    val_loader = DataLoader(VAL_DATASET, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=8, pin_memory=True)

    print(f"Validating Images:  {len(VAL_DATASET.samples)}")
    print(f"Training Images: {len(TRAIN_DATASET.samples)}")

    num_classes = len(set(label for _, label in TRAIN_DATASET.samples))
    print(f"Training on {num_classes} identities")

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
        "in_features": RESNET_PARAMS['dim'],
        "out_features": num_classes,
        "m": RESNET_PARAMS["arcface_m"],
        "s": RESNET_PARAMS["arcface_s"]
    }).to(DEVICE)

    trainable_backbone_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW([
        {"params": trainable_backbone_params, "lr": RESNET_PARAMS["lr_backbone"],
         "weight_decay": RESNET_PARAMS["weight_decay"]},
        {"params": arcface_loss.parameters(), "lr": RESNET_PARAMS["lr_arcface"],
         "weight_decay": RESNET_PARAMS["weight_decay"]}
    ])

    criterion = nn.CrossEntropyLoss(label_smoothing=RESNET_PARAMS["label_smooth"])
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=RESNET_PARAMS["warmup_epochs"]
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS - RESNET_PARAMS["warmup_epochs"], eta_min=1e-6
            )
        ],
        milestones=[RESNET_PARAMS["warmup_epochs"]]
    )

    # ---------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------
    patience = 10
    no_improve_count = 0
    best_val_acc = 0.0
    train_losses, train_accs, val_accs = [], [], []

    for epoch in range(NUM_EPOCHS):
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

        # Checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve_count = 0
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "arcface": arcface_loss.state_dict(),
                "val_acc": val_acc,
            }, RESNET_WEIGHTS)
        else:
            no_improve_count += 1

        print(
            f"Epoch {epoch + 1:02d}/{NUM_EPOCHS:02d} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Train Acc: {acc:.1f}% | "
            f"Val Acc: {val_acc:.1f}% | "
            f"LR backbone: {optimizer.param_groups[0]['lr']:.6f} | "
            f"LR arcface: {optimizer.param_groups[1]['lr']:.6f}"
        )

        if no_improve_count >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. Best val acc: {best_val_acc:.1f}%")
            break

    plot_curves(
        train_losses=train_losses,
        train_accs=train_accs,
        val_accs=val_accs,
        model_name=f"InceptionResnetV1_{NUM_EPOCHS}",
        save_dir=os.path.join(EVALUATION_DIR, "inception_resnet_v1/plots")
    )


if __name__ == "__main__":
    train_resnet()
