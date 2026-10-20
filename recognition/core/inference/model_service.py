from abc import ABC, abstractmethod
from pathlib import Path

import torch

from recognition.config import DEVICE
from recognition.core.utils.model_loader import (
    ModelLoader,
    ModelRegistry,
)


class ModelServiceAbs(ABC):
    @abstractmethod
    def load(self, weight_path: Path):
        pass


# ========Model Service Loading============================

class ViTModelService(ModelServiceAbs):
    def load(self, weight_path: Path):
        model = ModelLoader.load(ModelRegistry.ViTFace)
        arcface = ModelLoader.load(ModelRegistry.ArcFace)
        checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        arcface.load_state_dict(checkpoint['arcface'])
        model.eval()
        arcface.eval()
        return model.to(DEVICE)


class ResnetModelService(ModelServiceAbs):
    def load(self, weight_path: Path):
        model = ModelLoader.load(ModelRegistry.Inception_resnet)
        checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        model.eval()
        return model.to(DEVICE)


# =============================================================

class ModelWeightService:
    __models = {
        "inception_resnet_v1": ResnetModelService,
        "vit": ViTModelService
    }

    @classmethod
    def load(cls, model_name: str, weight_path: Path | str):
        model_class = cls.__models.get(model_name)
        if not model_class:
            raise ValueError(f"Model '{model_name}' is not supported. Supported models: {list(cls.__models.keys())}")
        return model_class().load(weight_path)
