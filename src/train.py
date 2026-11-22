"""Script de treinamento e avaliação da CNN construída do zero."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

# Permite executar tanto `python src/train.py` quanto `python -m src.train`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import EsteiraCNN
from src.preprocess import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IMAGE_SIZE,
    get_eval_transform,
    get_train_transform,
)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def seed_everything(seed: int = 42) -> None:
    """Fixa sementes para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GeometricShapesDataset(Dataset):
    """Dataset PyTorch a partir de listas de caminhos de imagem e rótulos."""

    def __init__(self, image_paths: List[Path], labels: List[int], transform=None) -> None:
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB")
            if self.transform is not None:
                tensor = self.transform(rgb_img)
            else:
                tensor = get_eval_transform()(rgb_img)

        return tensor, torch.tensor(label, dtype=torch.float32)


def load_dataset_paths(data_dir: Path, required: bool = True) -> Tuple[List[Path], List[int]]:
    """Carrega caminhos das imagens mapeando 0 = piramide e 1 = cubo."""
    image_paths: List[Path] = []
    labels: List[int] = []

    for class_name, class_idx in CLASS_TO_IDX.items():
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            if required:
                raise FileNotFoundError(
                    f"Pasta da classe '{class_name}' não encontrada em: {class_dir}"
                )
            continue

        files = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        )
        if not files and required:
            raise ValueError(f"Nenhuma imagem válida encontrada em: {class_dir}")

        image_paths.extend(files)
        labels.extend([class_idx] * len(files))

    return image_paths, labels


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> Tuple[float, float]:
    """Executa uma época de treino (se optimizer != None) ou avaliação."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_train):
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(inputs)
            loss = criterion(logits, targets)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == targets).sum().item()
            total += batch_size

    avg_loss = running_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def evaluate_on_test(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """Avalia o melhor modelo salvo no conjunto de teste (15%)."""
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long().cpu().tolist()

            y_pred.extend(preds)
            y_true.extend(targets.long().tolist())

    acc = accuracy_score(y_true, y_pred)
    target_names = [IDX_TO_CLASS[0], IDX_TO_CLASS[1]]

    print("\n" + "=" * 60)
    print("AVALIAÇÃO FINAL NO CONJUNTO DE TESTE (15%)")
    print("=" * 60)
    print(f"Acurácia: {acc:.4f} ({acc * 100:.2f}%)\n")
    print("Relatório de Classificação (Precisão, Recall, F1-Score):")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=target_names,
            digits=4,
            zero_division=0,
        )
    )
    print("Matriz de Confusão (linhas = real, colunas = predito):")
    print(f"Ordem das classes: [0={IDX_TO_CLASS[0]}, 1={IDX_TO_CLASS[1]}]")
    print(confusion_matrix(y_true, y_pred, labels=[0, 1]))
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina a CNN do zero para classificação de pirâmide (0) vs cubo (1)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Diretório contendo as pastas 'piramide' e 'cubo'.",
    )
    parser.add_argument(
        "--synthetic-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "synthetic",
        help="Diretório opcional com imagens sintéticas (usadas APENAS no treino).",
    )
    parser.add_argument(
        "--no-synthetic",
        action="store_true",
        help="Desabilita o uso de imagens sintéticas no conjunto de treino.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=PROJECT_ROOT / "model" / "cnn_esteira.pt",
        help="Caminho de saída para os melhores pesos treinados.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas.")
    parser.add_argument("--batch-size", type=int, default=8, help="Tamanho do batch.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taxa de aprendizado do Adam.")
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE, help="Resolução quadrada de entrada.")
    parser.add_argument("--seed", type=int, default=42, help="Semente aleatória.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizado: {device}")

    image_paths, labels = load_dataset_paths(args.data_dir)
    total_samples = len(image_paths)
    count_0 = sum(1 for y in labels if y == 0)
    count_1 = sum(1 for y in labels if y == 1)
    print(
        f"Total de imagens: {total_samples} "
        f"(0={IDX_TO_CLASS[0]}: {count_0}, 1={IDX_TO_CLASS[1]}: {count_1})"
    )

    # Divisão estratificada exata: 70% treino, 15% validação, 15% teste
    x_train, x_temp, y_train, y_temp = train_test_split(
        image_paths,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=args.seed,
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=args.seed,
    )

    print(
        f"Split estratificado (imagens reais) -> Treino: {len(x_train)} ({len(x_train)/total_samples:.0%}) | "
        f"Validação: {len(x_val)} ({len(x_val)/total_samples:.0%}) | "
        f"Teste: {len(x_test)} ({len(x_test)/total_samples:.0%})"
    )

    # Adiciona imagens sintéticas EXCLUSIVAMENTE ao treino (evita data leakage na validação/teste)
    n_real_train = len(x_train)
    real_y_train = list(y_train)
    synth_paths: List[Path] = []
    synth_labels: List[int] = []

    if not args.no_synthetic and args.synthetic_dir.is_dir():
        synth_paths, synth_labels = load_dataset_paths(args.synthetic_dir, required=False)
        if synth_paths:
            synth_0 = sum(1 for y in synth_labels if y == 0)
            synth_1 = sum(1 for y in synth_labels if y == 1)
            x_train = list(x_train) + synth_paths
            y_train = list(y_train) + synth_labels
            print(
                f"Imagens sintéticas adicionadas APENAS ao treino: +{len(synth_paths)} "
                f"(0={IDX_TO_CLASS[0]}: {synth_0}, 1={IDX_TO_CLASS[1]}: {synth_1}) | "
                f"Total treino: {len(x_train)}"
            )

    train_dataset = GeometricShapesDataset(
        x_train, y_train, transform=get_train_transform(args.image_size)
    )
    val_dataset = GeometricShapesDataset(
        x_val, y_val, transform=get_eval_transform(args.image_size)
    )
    test_dataset = GeometricShapesDataset(
        x_test, y_test, transform=get_eval_transform(args.image_size)
    )

    # Balanceia amostragem: 50% fotos reais (25% pirâmide + 25% cubo) e 50% sintéticas
    if synth_paths:
        real_c0 = max(sum(1 for y in real_y_train if y == 0), 1)
        real_c1 = max(sum(1 for y in real_y_train if y == 1), 1)
        syn_c0 = max(sum(1 for y in synth_labels if y == 0), 1)
        syn_c1 = max(sum(1 for y in synth_labels if y == 1), 1)

        sample_weights = []
        for idx, y in enumerate(y_train):
            if idx < n_real_train:
                sample_weights.append(0.25 / (real_c0 if y == 0 else real_c1))
            else:
                sample_weights.append(0.25 / (syn_c0 if y == 0 else syn_c1))

        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(y_train),
            replacement=True,
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = EsteiraCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = 0

    print("\nIniciando treinamento...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        improved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "image_size": args.image_size,
                    "class_to_idx": CLASS_TO_IDX,
                    "idx_to_class": IDX_TO_CLASS,
                    "val_loss": best_val_loss,
                    "val_acc": best_val_acc,
                    "epoch": best_epoch,
                },
                args.model_path,
            )
            improved = " *"

        print(
            f"Época [{epoch:02d}/{args.epochs:02d}] | "
            f"Treino Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}{improved}"
        )

    print(
        f"\nMelhor modelo salvo em '{args.model_path}' "
        f"(época {best_epoch}, val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.4f})."
    )

    # Carrega os melhores pesos salvos e avalia no conjunto de teste (15%)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    evaluate_on_test(model, test_loader, device)


if __name__ == "__main__":
    main()
