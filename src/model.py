"""Arquitetura da Rede Neural Convolucional (CNN) construída 100% do zero em PyTorch."""

import torch
import torch.nn as nn


class EsteiraCNN(nn.Module):
    """CNN customizada para classificação de peças da esteira (pirâmide, cubo e outros).

    Arquitetura:
    - 3 blocos convolucionais: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
    - AdaptiveAvgPool2d para independência de resolução de entrada
    - Cabeça classificadora totalmente conectada com Dropout e saída configurável
      (3 classes com CrossEntropyLoss por padrão).
    """

    def __init__(self, num_classes: int = 3, dropout_rate: float = 0.25) -> None:
        super().__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # Bloco 1: 3 -> 32 canais
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloco 2: 32 -> 64 canais
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloco 3: 64 -> 128 canais
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Normaliza o mapa espacial para 4x4 (compatível com 64x64 ou 128x128)
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(128 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        if self.num_classes == 1:
            return logits.squeeze(1)
        return logits
