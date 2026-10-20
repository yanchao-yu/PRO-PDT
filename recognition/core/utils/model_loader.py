from enum import Enum

from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1

from recognition.config import GPU_ID, DEVICE, VIT_PARAMS, ARCFACE_PARAMS
from recognition.core.models.ViT_face import ViT_face, ArcFace


class ModelRegistry(str, Enum):
    ArcFace = "ArcFace"
    ViTFace = "ViTFace"
    Inception_resnet = "Inception_resnet"


class ModelLoader:

    @classmethod
    def load(cls, model_name : ModelRegistry, overrides=None):
        overrides = overrides or {}

        match model_name:
            case ModelRegistry.ViTFace:
                params = {
                    **VIT_PARAMS,
                    **overrides
                }
                return ViT_face(**params).to(DEVICE)
            case ModelRegistry.ArcFace:
                params = {
                    **ARCFACE_PARAMS,
                    **overrides
                }
                return ArcFace(
                    device_id=GPU_ID,
                    **params
                ).to(DEVICE)
            case ModelRegistry.Inception_resnet:
                return InceptionResnetV1(pretrained=None)
            case _:
                raise Exception(f"Model {model_name} not supported")



