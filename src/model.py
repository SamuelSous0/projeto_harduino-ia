"""Arquitetura da Rede Neural Convolucional (CNN) construída 100% do zero em PyTorch."""

import torch
import torch.nn as nn


class EsteiraCNN(nn.Module):
    """CNN customizada para classificação binária (pirâmide vs cubo).

    Arquitetura:
    - 3 blocos convolucionais: Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
    - AdaptiveAvgPool2d para independência de resolução de entrada
    - Cabeça classificadora totalmente conectada com Dropout e saída de 1 logit
      para uso com BCEWithLogitsLoss.
    """

    def __init__(self, dropout_rate: float = 0.35) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Bloco 1: 3 -> 16 canais
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloco 2: 16 -> 32 canais
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Bloco 3: 32 -> 64 canais
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Normaliza o mapa espacial para 4x4 (compatível com 64x64 ou 128x128)
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.7),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits.squeeze(1)
