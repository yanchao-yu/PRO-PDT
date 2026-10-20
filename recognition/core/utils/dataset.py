import os
from dotenv import load_dotenv
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from recognition.config import VGGFACE_DATASET_ROOT
from recognition.core.inference.embedding_service import EmbeddingService

load_dotenv()


class MyDataset(Dataset):
    def __init__(self, root, split="train", val_split=0.2):
        """
        Args:
            root (str): Path to the VGGFace2 root directory.
            split (str): 'train', 'val', or 'test'.
            val_split (float): Fraction of train folder to reserve for validation (default 0.2).
        """
        self.samples = []
        self.split = split.lower()

        # 1. Image Transforms
        if self.split == "train":
            self.transform = EmbeddingService.transform(extra_transforms=[
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomRotation(10)
            ])
        else:  # Validation & Test splits (no augmentation)
            self.transform = EmbeddingService.transform()

        # 2. Determine target directory based on the requested split
        # train & val both load from root/train; test loads from root/val
        folder_name = "val" if self.split == "test" else "train"
        data_path = os.path.join(root, folder_name)

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Directory not found: {data_path}")

        identities = sorted([
            d for d in os.listdir(data_path)
            if os.path.isdir(os.path.join(data_path, d)) and not d.startswith(".")
        ])

        # 3. Populate samples list
        for label, identity in enumerate(identities):
            id_path = os.path.join(data_path, identity)
            images = sorted([
                f for f in os.listdir(id_path)
                if f.lower().endswith((".jpg", ".png", ".jpeg"))
            ])

            if not images:
                continue

            if self.split == "train":
                split_idx = int(len(images) * (1 - val_split))
                selected = images[:split_idx]  # First 80% of root/train
            elif self.split == "val":
                split_idx = int(len(images) * (1 - val_split))
                selected = images[split_idx:]  # Last 20% of root/train
            elif self.split == "test":
                selected = images  # 100% of root/val
            else:
                raise ValueError(f"Unknown split: '{split}'. Expected 'train', 'val', or 'test'.")

            for img_file in selected:
                path = os.path.join(id_path, img_file)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


# 1. Initialize Datasets

dataset = os.environ.get("VGGFACE_DATASET_ROOT")

if dataset is None:
    print("VGGFACE_DATASET_ROOT environment variable is not set. Using the default dataset.")
    dataset = VGGFACE_DATASET_ROOT

TRAIN_DATASET = MyDataset(root=dataset, split="train", val_split=0.2)
VAL_DATASET = MyDataset(root=dataset, split="val", val_split=0.2)
TEST_DATASET = MyDataset(root=dataset, split="test")
