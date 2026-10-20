import torch
import torch.nn.functional as F

from PIL import Image
from facenet_pytorch import MTCNN
from torchvision import transforms


class EmbeddingService:
    face_detector = MTCNN(
        image_size=112,
        margin=20,
        keep_all=False,  # Returns single face
        select_largest=True,  # Forces choice of closest/largest face
        selection_method="largest",
        device="cpu"
    )

    @classmethod
    def transform(
            cls,
            resize=(112, 112),
            mean=None,
            std=None,
            extra_transforms=None,
    ):
        mean = mean or [0.5, 0.5, 0.5]
        std = std or [0.5, 0.5, 0.5]
        extra_transforms = extra_transforms or []

        transform_list = [
            transforms.Resize(resize),
            *extra_transforms,
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

        return transforms.Compose(transform_list)

    @classmethod
    def get_embedding(cls, path, model):
        from recognition.config import DEVICE

        try:
            img = Image.open(path).convert("RGB")
        except FileNotFoundError as e:
            print(f"File {path} does not exist!")
        face = cls.face_detector(img)
        if face is None:
            print("No face detected")
            return None
        # transform = cls.transform()
        img = face.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = model(img)
        embedding = F.normalize(embedding, dim=1)
        return embedding
