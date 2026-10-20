import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from recognition.core.utils.model_loader import ModelLoader, ModelRegistry

from recognition.config import DEVICE, BATCH_SIZE, VIT_WEIGHTS
from recognition.core.utils.dataset import TEST_DATASET

# ── CONFIG ────────────────────────────────────────────────────────────
MODEL_NAME = "ViT_Face_Transformer"
num_classes = 480
SAVE_DIR = "core/evaluation/vision_transformer/results"

# ── LOAD VAL DATA ─────────────────────────────────────────────────────
print(f'dataset size: {len(TEST_DATASET.samples)}')
test_loader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=8, pin_memory=True)

# ── LOAD MODEL ────────────────────────────────────────────────────────

model = ModelLoader.load(
    ModelRegistry.ViTFace,
    overrides={"num_class": num_classes}
).to(DEVICE)

arcface_loss = ModelLoader.load(ModelRegistry.ArcFace, overrides={
    "in_features": 512,
    "out_features": num_classes,
    "m": 0.2907365733481595,  # reduced from 0.5 — easier convergence for fine-tuning
    "s": 48.070429215282175  # reduced from 64.0 — reduces loss scale
}).to(DEVICE)

checkpoint = torch.load(VIT_WEIGHTS, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model'])
arcface_loss.load_state_dict(checkpoint['arcface'])
model = model.to(DEVICE)
model.eval()
arcface_loss.eval()

# ── EXTRACT ALL EMBEDDINGS ────────────────────────────────────────────
all_embeddings = []
all_labels = []

with torch.no_grad():
    print('Loading Embeddings...')

    for images, labels in tqdm(test_loader):
        images = images.to(DEVICE, non_blocking=True)
        with torch.amp.autocast(device_type=DEVICE.type,
                                enabled=(DEVICE.type == "cuda")):
            embeddings = model(images)
            embeddings = F.normalize(embeddings, dim=1)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)
    print("Ended Embedding")
all_embeddings = torch.cat(all_embeddings)  # [N, 512]
all_labels = torch.cat(all_labels)  # [N]
similarity_matrix = torch.mm(all_embeddings, all_embeddings.T)

print(f"Extracted embeddings: {all_embeddings.shape}")
print(f"Labels: {all_labels.shape}")


def compute_cmc(similarity_matrix, labels, ranks=[1, 5]):
    num_queries = similarity_matrix.shape[0]
    correct_at_rank = {r: 0 for r in ranks}

    for i in range(num_queries):
        scores = similarity_matrix[i].clone()

        # remove the query itself
        scores[i] = -1

        # sort by descending similarity
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices]

        # check if correct label appears within each rank
        query_label = labels[i]
        for r in ranks:
            if query_label in sorted_labels[:r]:
                correct_at_rank[r] += 1
    results = {r: correct_at_rank[r] / num_queries * 100 for r in ranks}
    return results


def compute_map(similarity_matrix, labels):
    num_queries = similarity_matrix.shape[0]
    average_precisions = []

    for i in range(num_queries):
        scores = similarity_matrix[i].clone()
        scores[i] = -1

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices]
        query_label = labels[i]

        relevant = (sorted_labels == query_label).float()

        total_relevant = relevant.sum().item()
        if total_relevant == 0:
            continue

        # cumulative sum of relevant items at each position
        cumulative_hits = torch.cumsum(relevant, dim=0)

        # position indices starting from 1
        positions = torch.arange(1, len(relevant) + 1).float()

        # precision at each position
        precision_at_k = cumulative_hits / positions

        # only keep precision values where the item is relevant
        ap = (precision_at_k * relevant).sum().item() / total_relevant
        average_precisions.append(ap)

    return sum(average_precisions) / len(average_precisions) * 100


def compute_auc(similarity_matrix, labels):
    num_queries = similarity_matrix.shape[0]
    scores_list = []
    binary_labels = []

    # sample a subset of pairs to keep it manageable
    # checking all N^2 pairs would be 35476^2 = 1.2 billion
    num_samples = 100000
    indices = torch.randperm(num_queries)[:1000]

    for i in indices:
        for j in indices:
            if i == j:
                continue
            scores_list.append(similarity_matrix[i][j].item())
            binary_labels.append(
                1 if labels[i] == labels[j] else 0
            )

    return roc_auc_score(binary_labels, scores_list) * 100


print("Computing similarity matrix...")
sim_matrix = torch.mm(all_embeddings, all_embeddings.T)

print("Computing CMC...")
cmc = compute_cmc(sim_matrix, all_labels)

print("Computing mAP...")
map_score = compute_map(sim_matrix, all_labels)

print("Computing AUC...")
auc = compute_auc(sim_matrix, all_labels)

print(f"\nRank-1 : {cmc[1]:.2f}%")
print(f"Rank-5 : {cmc[5]:.2f}%")
print(f"mAP    : {map_score:.2f}%")
print(f"AUC    : {auc:.2f}%")
