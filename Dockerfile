FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    PORT=7860 \
    CONFIDENCE_THRESHOLD=0.75

WORKDIR /app

# Instala apenas as dependências de inferência da API (sem scikit-learn, mais leve para 512 MB RAM no Render)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copia apenas o código da API/modelo e os pesos treinados (data/ fica fora via .dockerignore)
COPY src/ ./src/
COPY model/ ./model/

EXPOSE 7860

# Compatível com Render (variável PORT) e Hugging Face Spaces (porta 7860)
CMD ["sh", "-c", "uvicorn src.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
