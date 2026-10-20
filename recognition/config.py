import enum
from pathlib import Path
import torch


# ============================================================
# Project Model
# ============================================================

MODEL = "inception_resnet_v1"

# ============================================================
# Project Root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# Dataset Paths
# ============================================================

VGGFACE_DATASET_ROOT = Path("/users/40771994/sharedscratch/datasets/vggface2")


# ============================================================
# Database Paths
# ============================================================

DATABASE_PATH = (PROJECT_ROOT / "faces.db")

# ============================================================
# Model and Output Paths
# ============================================================

MODEL_DIR = Path(
    "/users/40771994/localscratch/models"
)

OUTPUT_DIR = Path(
    "/users/40771994/localscratch/output"
)

EVALUATION_DIR = PROJECT_ROOT / "core" / "evaluation"

VIT_WEIGHTS = PROJECT_ROOT / "core" / "weights" / "vit_face_best.pth"
RESNET_WEIGHTS = PROJECT_ROOT / "core" / "weights" / "resnet_best.pth"
PRETRAINED_VIT_WEIGHTS = PROJECT_ROOT / "core" / "weights" / "external" / "pretrained_vit_face.pth"

# ============================================================
# Device
# ============================================================

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    GPU_ID = [0]
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    GPU_ID = []
else:
    DEVICE = torch.device("cpu")
    GPU_ID = []

# ============================================================
# ViT Face Configuration
# ============================================================

#
# VIT_PARAMS = {
#     "loss_type": "arcface",
#     "patch_size": 8,
#     "depth": 6,
#     "heads": 8,
#     "mlp_dim": 2048,
#     "dropout": 0.1,
#     "emb_dropout": 0.1,
#     "num_class": 480,
#     "image_size": 112,
#     "dim": 512,
#     "GPU_ID": GPU_ID
# }

VIT_PARAMS = {
    "loss_type": "arcface",
    "GPU_ID": [0],
    "num_class": 480,
    "image_size": 112,
    "patch_size": 8,
    "dim": 512,
    "depth": 20,  # changed from 6 to match pretrained
    "heads": 8,
    "mlp_dim": 2048,
    "dropout": 0.1,
    "emb_dropout": 0.1,
}

VIT_CUSTOM_PARAMS = {
    "lr_backbone": 0.0009343088526105685,
    "lr_arcface": 0.0008297428450517266,
    "weight_decay": 0.00032825958642964943,
    "label_smooth": 0.01202998986011552,
    "arcface_m": 0.2907365733481595,
    "arcface_s": 48.070429215282175 ,
    "warmup_epochs": 4
}



# ============================================================
# ArcFace Configuration
# ============================================================


ARCFACE_PARAMS = {
    "m": 0.3,
    "s": 32.0,
    "in_features": 512,
    "out_features": 480,
}
# ============================================================
# Inception_resnet Configuration
# ============================================================

RESNET_PARAMS = {
    "dim": 512,
    "lr_backbone": 4.0427095982153816e-05,
    "lr_arcface": 0.00013472913464689117,
    "weight_decay": 0.00011769689679347525,
    "label_smooth": 0.08511383579311559,
    "arcface_m": 0.20425466140056883,
    "arcface_s": 53.094149783741244,
    "warmup_epochs": 4
}





# ============================================================
# Training Configuration
# ============================================================

BATCH_SIZE = 128
NUM_EPOCHS = 100
MAX_LR = 1e-3
SIMILARITY_THRESHOLD = 0.70
LOW_THRESHOLD=0.50