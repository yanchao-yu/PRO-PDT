import os
from recognition.config import DEVICE, RESNET_WEIGHTS, BATCH_SIZE, EVALUATION_DIR
from recognition.core.utils.dataset import VAL_DATASET
import torch
from torch.utils.data import DataLoader
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn as nn

from recognition.core.utils.evaluate_saver import save_evaluation

val_loader = DataLoader(VAL_DATASET, batch_size=BATCH_SIZE, shuffle=True)
num_classes = len(set(label for _, label in VAL_DATASET.samples))
model = InceptionResnetV1(pretrained=None)
model.logits = nn.Linear(512, num_classes)
model.load_state_dict(torch.load(RESNET_WEIGHTS, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        all_preds.append(predictions)
        all_labels.append(labels)

# Convert to flat lists for sklearn
all_preds = torch.cat(all_preds).cpu().numpy()
all_labels = torch.cat(all_labels).cpu().numpy()

save_evaluation(
    model_name="inception_resnet",
    dataset="VGGFace2",
    split="val",
    save_dir=os.path.join(EVALUATION_DIR, "inception_resnet_v1/results"),
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
        "weights_file": RESNET_WEIGHTS,
        "epochs_trained": 20,
        "batch_size": BATCH_SIZE
    }
)