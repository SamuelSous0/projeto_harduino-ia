# Esteira Seletora com CNN (PyTorch do Zero) + FastAPI + ESP32

Pipeline completo de classificação de imagens utilizando uma **Rede Neural Convolucional (CNN) construída 100% do zero em PyTorch** (sem modelos pré-treinados ou *transfer learning*) e uma **API FastAPI** projetada para ser consumida por um **ESP32** em uma esteira seletora.

## Estrutura do Projeto

```text
.
├── data/
│   ├── piramide/          # Classe 0 (imagens reais - triângulo / pirâmide)
│   ├── cubo/              # Classe 1 (imagens reais - quadrado / cubo)
│   └── synthetic/         # Imagens sintéticas geradas por IA (usadas APENAS no treino)
│       ├── piramide/
│       └── cubo/
├── model/
│   └── cnn_esteira.pt     # Gerado após o treinamento
├── src/
│   ├── __init__.py
│   ├── preprocess.py      # Pré-processamento unificado (treino e API)
│   ├── model.py           # Arquitetura da CNN customizada em PyTorch
│   ├── train.py           # Split 70/15/15, treinamento e avaliação no teste
│   └── main.py            # API FastAPI (GET /health e POST /predict)
├── Dockerfile             # Deploy gratuito da API (Render / Hugging Face Spaces)
├── render.yaml            # Blueprint 1-clique para Render (Free + /health)
├── .dockerignore          # Exclui data/ da imagem de produção
├── train_colab.ipynb      # Notebook para treinar gratuitamente no Google Colab
├── requirements.txt       # Dependências completas (treino + API)
├── requirements-api.txt   # Dependências enxutas só para produção (menor RAM)
└── README.md
```

## 1. Instalação das Dependências (CPU)

Crie e ative um ambiente virtual Python (recomendado Python 3.10+) e instale as dependências:

```bash
python -m venv .venv
```

No **Windows (PowerShell)**:
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

No **Linux/macOS**:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Treinamento e Avaliação da CNN

Execute o script de treinamento:

```bash
python src/train.py
```

Parâmetros opcionais:
```bash
python src/train.py --epochs 40 --batch-size 8 --lr 0.001 --image-size 64
# Para treinar sem as imagens sintéticas:
python src/train.py --no-synthetic
```

O script realiza automaticamente:
1. Carregamento das imagens reais de `data/piramide` (`0`) e `data/cubo` (`1`).
2. Divisão estratificada exata das imagens reais em **70% treino, 15% validação e 15% teste**.
3. Inclusão das imagens sintéticas de `data/synthetic/` **exclusivamente no conjunto de treino** (mantendo validação e teste 100% com fotos reais, sem *data leakage*).
4. Treinamento com `BCEWithLogitsLoss` e otimizador `Adam`.
5. Salvamento dos melhores pesos em `model/cnn_esteira.pt` com base na validação.
6. Avaliação final no conjunto de teste (15%), exibindo **Acurácia, Precisão, Recall, F1-Score (`classification_report`)** e **Matriz de Confusão (`confusion_matrix`)**.

## 3. Subindo a API para o ESP32

Inicie o servidor com `--host 0.0.0.0` para que o ESP32 consiga acessá-lo na mesma rede Wi-Fi:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Para ajustar o limiar de confiança da classe `"nao_reconhecido"` (padrão `0.75`):

**Windows (PowerShell):**
```powershell
$env:CONFIDENCE_THRESHOLD="0.80"
uvicorn src.main:app --host 0.0.0.0 --port 8000
```

**Linux/macOS:**
```bash
CONFIDENCE_THRESHOLD=0.80 uvicorn src.main:app --host 0.0.0.0 --port 8000
```

## 4. Endpoints da API

### `GET /health`
Verifica se a API está ativa e se `model/cnn_esteira.pt` foi carregado na inicialização:

```bash
curl http://localhost:8000/health
```

### `POST /predict`
Recebe uma imagem via `multipart/form-data` (campo `file`):

```bash
curl -X POST http://localhost:8000/predict -F "file=@data/cubo/CUBO1.jpeg"
```

#### Exemplos de Resposta JSON para o ESP32

- **Pirâmide (`y = 0`):**
```json
{
  "classe": "piramide",
  "y": 0,
  "acao": "desviar_piramide",
  "confianca": 0.9412
}
```

- **Cubo (`y = 1`):**
```json
{
  "classe": "cubo",
  "y": 1,
  "acao": "desviar_cubo",
  "confianca": 0.9587
}
```

- **Não reconhecido (`y = -1`, ex.: esfera ou objeto desconhecido):**
Quando `confianca < CONFIDENCE_THRESHOLD`, a API sinaliza para deixar a peça passar reto na esteira:
```json
{
  "classe": "nao_reconhecido",
  "y": -1,
  "acao": "passar_reto",
  "confianca": 0.6143
}
```

> **Dica de Engenharia de ML para objetos desconhecidos (ex.: esfera):**
> Como a CNN binária foi treinada apenas com fotos de **pirâmide** e **cubo**, o limiar `CONFIDENCE_THRESHOLD` ajuda a rejeitar entradas ambíguas. Em testes reais na esteira, ajuste `CONFIDENCE_THRESHOLD` (por exemplo, `0.80` ou `0.85`). Se futuramente quiser rejeição ainda mais robusta contra esferas ou esteira vazia, basta coletar ~20–30 fotos de outros objetos em uma terceira pasta (`data/outros`) e evoluir a saída para classificação de 3 classes (`CrossEntropyLoss`).

## 5. Treino e Hospedagem Gratuitos na Nuvem (ESP32)

> **Importante:** A API hospedada **não precisa da pasta `data/`** (apenas `src/` + `model/cnn_esteira.pt`, que pesa ~300 KB). O `.dockerignore` já exclui `data/` automaticamente da imagem Docker de produção.

### Passo A — Treinar gratuitamente no Google Colab
1. Suba este repositório (incluindo `data/`, que tem apenas `21,6 MB`) para o **GitHub** (ou suba a pasta `data/` para o **Google Drive**).
2. Abra o arquivo `train_colab.ipynb` no [Google Colab](https://colab.research.google.com/).
3. Configure `ORIGEM_DADOS` (`"github"` ou `"drive"`) e execute todas as células.
4. O Colab baixará automaticamente o arquivo `cnn_esteira.pt`. Coloque-o em `model/cnn_esteira.pt` e faça *commit/push*.

### Passo B — Hospedar a API gratuitamente no Render (ou Hugging Face Spaces)

**Opção 1 — Render (Free):**
1. Suba o repositório no GitHub (incluindo `model/cnn_esteira.pt`).
2. No [Render](https://render.com/), clique em **New → Blueprint** (detecta o `render.yaml` automaticamente) ou **New → Web Service** escolhendo **Docker** e plano **Free**.
3. O `Dockerfile` usa `requirements-api.txt` (sem `scikit-learn`) e `OMP_NUM_THREADS=1` para caber com folga nos 512 MB de RAM.
4. No ESP32, faça um `GET /health` ao ligar a esteira para acordar a instância caso ela tenha hibernado após 15 min sem uso.

**Opção 2 — Hugging Face Spaces (Free — 16 GB RAM):**
1. Crie um **New Space** em [Hugging Face](https://huggingface.co/) escolhendo **Docker (Blank)** e **CPU basic (Free)**.
2. Envie o repositório — o mesmo `Dockerfile` sobe automaticamente na porta `7860` (`https://<seu-user>-<seu-space>.hf.space`).

## 6. Integração com o ESP32

No firmware do ESP32 (ex.: ESP32-CAM usando `WiFiClientSecure` / `HTTPClient` + `ArduinoJson`):
- Envie o frame JPEG via `POST https://<URL_DA_SUA_API>/predict` com `Content-Type: multipart/form-data` e campo `file`.
- Leia o campo inteiro `y` da resposta JSON:
  - `0`: acionar atuador/servo da **pirâmide**
  - `1`: acionar atuador/servo do **cubo**
  - `-1`: **não acionar nenhum atuador** (deixar passar reto na esteira)
