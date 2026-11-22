Atue como um engenheiro de Machine Learning e Back-end Python.

Neste repositório existem inicialmente apenas **duas pastas contendo as imagens das formas geométricas** (uma pasta para triângulo/pirâmide e outra para quadrado/cubo). Inspecione o diretório atual para identificar os nomes exatos dessas duas pastas antes de escrever o código.

### Objetivo do Projeto
Criar um pipeline completo de classificação binária de imagens usando uma **Rede Neural Convolucional (CNN) treinada 100% do zero em PyTorch** e expor o modelo treinado em uma **API Back-end com FastAPI** que será consumida por uma placa embarcada (Arduino/ESP32) em uma esteira seletora.

### Restrições Obrigatórias
- **NÃO usar modelos pré-treinados** (proibido `pretrained=True`, transfer learning, ResNet/MobileNet prontas ou YOLOv8). A arquitetura da CNN deve ser definida manualmente do zero em PyTorch.
- Usar **PyTorch** para a CNN, **scikit-learn** para o split estratificado e métricas de avaliação, e **FastAPI + Uvicorn** para o back-end.

### Estrutura de Arquivos a Criar
Organize o projeto de forma limpa:
- `requirements.txt` (usando dependências compatíveis com CPU)
- `src/preprocess.py`: transformações de imagem padronizadas (redimensionamento para `64x64` ou `128x128`, normalização e *data augmentation* apenas para o conjunto de treino). Deve ser reutilizado tanto no treino quanto na API para evitar divergência.
- `src/model.py`: classe da CNN customizada em PyTorch (2 a 3 blocos `Conv2d` + `BatchNorm2d` + `ReLU` + `MaxPool2d`, seguidos de camadas `Linear` para classificação binária).
- `src/train.py`: script completo de treinamento e avaliação:
  1. Carrega as imagens das duas pastas existentes mapeando `0` = triângulo/pirâmide e `1` = quadrado/cubo.
  2. Faz a divisão estratificada exata em **70% treino, 15% validação e 15% teste**.
  3. Treina com `BCEWithLogitsLoss` e otimizador `Adam`, acompanhando *loss* e acurácia de treino e validação a cada época.
  4. Salva os melhores pesos em `model/cnn_esteira.pt` com base na validação.
  5. Ao final, carrega o melhor modelo e avalia no conjunto de **teste (15%)**, imprimindo **Acurácia, Precisão, Recall, F1-Score (`classification_report`) e Matriz de Confusão (`confusion_matrix`)**.
- `src/main.py`: API FastAPI:
  - Carrega o modelo `model/cnn_esteira.pt` **uma única vez** na inicialização do servidor (lifespan).
  - `GET /health`: retorna status da API e se o modelo está carregado.
  - `POST /predict`: recebe uma imagem via `UploadFile` (`multipart/form-data`), aplica o mesmo pré-processamento de validação/teste, executa a inferência e retorna um JSON enxuto para a placa:
    ```json
    {
      "classe": "quadrado",
      "y": 1,
      "confianca": 0.95
    }
    ```
- `README.md`: instruções objetivas de como instalar as dependências, rodar o `train.py` e subir a API com `uvicorn`.

Antes de começar a criar os arquivos, verifique os nomes das duas pastas de imagens no workspace e confirme o mapeamento `0` (triângulo/pirâmide) e `1` (quadrado/cubo).