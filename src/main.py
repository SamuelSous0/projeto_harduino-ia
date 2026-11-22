"""API FastAPI para inferência em tempo real da esteira seletora com ESP32."""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from src.model import EsteiraCNN
from src.preprocess import (
    CLASS_TO_ACTION,
    IDX_TO_CLASS,
    IMAGE_SIZE,
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
    app.state.image_size = IMAGE_SIZE
    app.state.confidence_threshold = CONFIDENCE_THRESHOLD

    if MODEL_PATH.is_file():
        checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model = EsteiraCNN().to(device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            app.state.image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        app.state.model = model

    yield

    app.state.model = None


app = FastAPI(
    title="API Esteira Seletora - CNN PyTorch + ESP32",
    description=(
        "Classificação binária de peças (pirâmide vs cubo) com rejeição por limiar "
        "de confiança para objetos não reconhecidos (passar reto na esteira)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check():
    """Retorna o estado da API e se o modelo foi carregado corretamente."""
    modelo_carregado = app.state.model is not None
    return {
        "status": "ok" if modelo_carregado else "degraded",
        "modelo_carregado": modelo_carregado,
        "modelo_path": str(MODEL_PATH),
        "dispositivo": str(app.state.device),
        "confidence_threshold": app.state.confidence_threshold,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Recebe uma imagem enviada pelo ESP32 e retorna a classe, código e ação da esteira.

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
        logit = model(input_tensor)
        prob_cubo = torch.sigmoid(logit).item()

    if prob_cubo >= 0.5:
        raw_y = 1
        confianca = prob_cubo
    else:
        raw_y = 0
        confianca = 1.0 - prob_cubo

    if confianca < app.state.confidence_threshold:
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
