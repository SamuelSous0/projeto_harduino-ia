"""API FastAPI para inferência em tempo real da esteira seletora com ESP32."""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from src.model import EsteiraCNN
from src.preprocess import (
    CLASS_TO_ACTION,
    IDX_TO_CLASS,
    IMAGE_SIZE,
    OUTROS_CLASS_IDX,
    UNKNOWN_CLASS_IDX,
    UNKNOWN_CLASS_NAME,
    preprocess_pil_image,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "model" / "cnn_esteira.pt"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo PyTorch uma única vez na inicialização da API."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app.state.device = device
    app.state.model = None
    app.state.num_classes = 3
    app.state.image_size = IMAGE_SIZE
    app.state.confidence_threshold = CONFIDENCE_THRESHOLD

    if MODEL_PATH.is_file():
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            num_classes = int(checkpoint.get("num_classes", 1))
            model = EsteiraCNN(num_classes=num_classes).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            app.state.image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
            app.state.num_classes = num_classes
        else:
            model = EsteiraCNN(num_classes=1).to(device)
            model.load_state_dict(checkpoint)
            app.state.num_classes = 1
        model.eval()
        app.state.model = model

    yield

    app.state.model = None


app = FastAPI(
    title="API Esteira Seletora - CNN PyTorch + ESP32",
    description=(
        "Classificação de peças (pirâmide, cubo e não reconhecido/outros) com "
        "CNN construída do zero em PyTorch para controle de esteira com ESP32."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Permite chamadas diretas de páginas web / câmera no navegador (PC ou celular)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    """Retorna o estado da API e se o modelo foi carregado corretamente."""
    modelo_carregado = app.state.model is not None
    return {
        "status": "ok" if modelo_carregado else "degraded",
        "modelo_carregado": modelo_carregado,
        "num_classes": app.state.num_classes,
        "modelo_path": str(MODEL_PATH),
        "dispositivo": str(app.state.device),
        "confidence_threshold": app.state.confidence_threshold,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Recebe uma imagem e retorna a classe, código e ação da esteira.

    Retornos possíveis:
    - y = 0  -> classe: "piramide",        acao: "desviar_piramide"
    - y = 1  -> classe: "cubo",            acao: "desviar_cubo"
    - y = -1 -> classe: "nao_reconhecido", acao: "passar_reto"
    """
    model = app.state.model
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Modelo não carregado. Execute 'python src/train.py' para gerar "
                f"'{MODEL_PATH}' e reinicie a API."
            ),
        )

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Arquivo de imagem vazio.")

    try:
        image = Image.open(io.BytesIO(image_bytes))
        input_tensor = preprocess_pil_image(
            image, image_size=app.state.image_size
        ).to(app.state.device)
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400, detail="Arquivo enviado não é uma imagem válida."
        ) from exc

    with torch.no_grad():
        logits = model(input_tensor)

        if app.state.num_classes == 1:
            prob_cubo = torch.sigmoid(logits).item()
            if prob_cubo >= 0.5:
                raw_y, confianca = 1, prob_cubo
            else:
                raw_y, confianca = 0, 1.0 - prob_cubo
        else:
            probs = torch.softmax(logits, dim=1)[0]
            conf_tensor, idx_tensor = torch.max(probs, dim=0)
            raw_y = int(idx_tensor.item())
            confianca = float(conf_tensor.item())

    if raw_y == OUTROS_CLASS_IDX or confianca < app.state.confidence_threshold:
        return {
            "classe": UNKNOWN_CLASS_NAME,
            "y": UNKNOWN_CLASS_IDX,
            "acao": CLASS_TO_ACTION[UNKNOWN_CLASS_IDX],
            "confianca": round(confianca, 4),
        }

    return {
        "classe": IDX_TO_CLASS[raw_y],
        "y": raw_y,
        "acao": CLASS_TO_ACTION[raw_y],
        "confianca": round(confianca, 4),
    }
