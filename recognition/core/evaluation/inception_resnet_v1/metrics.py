import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from recognition.config import DEVICE, RESNET_WEIGHTS, RESNET_PARAMS
from recognition.core.utils.dataset import TEST_DATASET
from recognition.core.utils.model_loader import ModelLoader, ModelRegistry

# ── CONFIG ────────────────────────────────────────────────────────────
MODEL_NAME = "inception_resnet"
NUM_CLASSES = 480
BATCH_SIZE = 128
SAVE_DIR = "core/evaluation/inception_resnet_v1/results"

# ── LOAD MODEL ────────────────────────────────────────────────────────
model = InceptionResnetV1(pretrained=None, classify=False).to(DEVICE)

arcface_loss = ModelLoader.load(ModelRegistry.ArcFace, overrides={
    "in_features": RESNET_PARAMS['dim'],
    "out_features": NUM_CLASSES,
    "m": RESNET_PARAMS["arcface_m"],
    "s": RESNET_PARAMS["arcface_s"]
}).to(DEVICE)

checkpoint = torch.load(RESNET_WEIGHTS, map_location=DEVICE, weights_only=False)
model.load_state_dict(checkpoint['model'], strict=False)
arcface_loss.load_state_dict(checkpoint['arcface'])
model = model.to(DEVICE)
model.eval()
arcface_loss.eval()

# ── LOAD VAL DATA ─────────────────────────────────────────────────────
print("Loading Dataset...")
print(f'dataset size: {len(TEST_DATASET.samples)}')
test_loader = DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=8, pin_memory=True)

# ── EXTRACT ALL EMBEDDINGS ────────────────────────────────────────────
all_embeddings = []
all_labels = []

with torch.no_grad():
    print('Loading Embeddings...')

    for images, labels in test_loader:
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
