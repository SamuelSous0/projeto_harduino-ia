"""Pré-processamento padronizado de imagens compartilhado entre treino e API."""

from typing import Dict

import torch
from PIL import Image
from torchvision import transforms

IMAGE_SIZE: int = 128
NORMALIZE_MEAN = (0.5, 0.5, 0.5)
NORMALIZE_STD = (0.5, 0.5, 0.5)

# Mapeamento oficial de classes (compartilhado entre treino e API)
CLASS_TO_IDX: Dict[str, int] = {
    "piramide": 0,
    "cubo": 1,
}
IDX_TO_CLASS: Dict[int, str] = {v: k for k, v in CLASS_TO_IDX.items()}

UNKNOWN_CLASS_NAME: str = "nao_reconhecido"
UNKNOWN_CLASS_IDX: int = -1

CLASS_TO_ACTION: Dict[int, str] = {
    0: "desviar_piramide",
    1: "desviar_cubo",
    UNKNOWN_CLASS_IDX: "passar_reto",
}


def get_train_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Transformações com data augmentation usadas exclusivamente no treino."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
            ),
            transforms.RandomPerspective(distortion_scale=0.15, p=0.3),
            transforms.ColorJitter(
                brightness=0.25,
                contrast=0.25,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.RandomGrayscale(p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def get_eval_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Transformações determinísticas para validação, teste e inferência na API."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )


def preprocess_pil_image(
    image: Image.Image,
    image_size: int = IMAGE_SIZE,
) -> torch.Tensor:
    """Converte uma imagem PIL para RGB, aplica o transform de avaliação e adiciona dimensão de batch."""
    rgb_image = image.convert("RGB")
    tensor = get_eval_transform(image_size=image_size)(rgb_image)
    return tensor.unsqueeze(0)
