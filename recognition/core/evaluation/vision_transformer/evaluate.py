import os

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader

from recognition.config import DEVICE, VIT_WEIGHTS, BATCH_SIZE, EVALUATION_DIR
from recognition.core.utils.dataset import VAL_DATASET
from recognition.core.utils.evaluate_saver import save_evaluation
from recognition.core.utils.model_loader import ModelLoader, ModelRegistry

print(f"Using DEVICE: {DEVICE}")


val_loader = DataLoader(VAL_DATASET, batch_size=BATCH_SIZE, shuffle=True)

num_classes = len(set(label for _, label in VAL_DATASET.samples))
print(f"Validating on {num_classes} identities")


model = ModelLoader.load(ModelRegistry.ViTFace, overrides={"num_class": num_classes})
arcface_loss = ModelLoader.load(ModelRegistry.ArcFace)

checkpoint = torch.load(VIT_WEIGHTS, map_location=DEVICE)
model.load_state_dict(checkpoint['model'])
arcface_loss.load_state_dict(checkpoint['arcface'])
model = model.to(DEVICE)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        embeddings = model(images)
        logits = arcface_loss(embeddings, labels)
        _, predictions = torch.max(logits, 1)
        all_preds.append(predictions)
        all_labels.append(labels)


# Convert to flat lists for sklearn
all_preds = torch.cat(all_preds).cpu().numpy()
all_labels = torch.cat(all_labels).cpu().numpy()

save_evaluation(
    model_name="ViT_Face_Transformer",
    dataset="VGGFace2",
    split="val",
    save_dir=os.path.join(EVALUATION_DIR,"vision_transformer/results"),
    metrics={
        "accuracy": round(accuracy_score(all_labels, all_preds) * 100, 2),
        "precision": round(precision_score(all_labels, all_preds, average='macro', zero_division=0) * 100, 2),
        "recall": round(recall_score(all_labels, all_preds, average='macro', zero_division=0) * 100, 2),
        "f1": round(f1_score(all_labels, all_preds, average='macro', zero_division=0) * 100, 2),
    },
    extra_info={
        "num_classes": num_classes,
        "num_val_samples": len(all_labels),
        "DEVICE": str(DEVICE),
        "weights_file": VIT_WEIGHTS,
        "epochs_trained": 100,
        "batch_size": BATCH_SIZE
    }
)


