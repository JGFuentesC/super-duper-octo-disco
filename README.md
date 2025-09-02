# Super Duper Octo Disco - Machine Learning Repository

Este repositorio contiene notebooks y scripts de machine learning organizados por temas.

## Estructura del Repositorio

```
.
├── data/
│   ├── raw/                    # Datasets originales
│   │   ├── catsDogsMiniDataset/    # Dataset de gatos y perros
│   │   ├── coranBibliaCleanDataset.pkl  # Dataset limpio Corán-Biblia
│   │   ├── ludicoDataset.csv        # Dataset lúdico
│   │   ├── mnistDataset/            # Dataset MNIST
│   │   ├── reviewsDataset.xlsx      # Dataset de reseñas
│   │   └── titanicDataset.csv      # Dataset del Titanic
│   └── processed/              # Datasets procesados (futuro)
├── notebooks/                  # Jupyter notebooks organizados por tema
│   ├── cnn/                   # Convolutional Neural Networks
│   │   └── helloCnnNotebook.ipynb
│   ├── embeddings/            # Embeddings y GenAI
│   │   ├── helloEmbeddingNotebook.ipynb
│   │   └── helloGenAiNotebook.ipynb
│   ├── nlp/                   # Natural Language Processing
│   │   └── helloNlpNotebook.ipynb
│   ├── titanic/               # Análisis del dataset Titanic
│   │   └── titanicLearningNotebook.ipynb
│   ├── vae/                   # Variational Autoencoders
│   │   ├── vaeCatsModifiedNotebook.ipynb
│   │   ├── vaeCatsNotebook.ipynb
│   │   └── vaeMnistNotebook.ipynb
│   └── visualization/         # Visualizaciones
│       └── cnnVisualizationNotebook.ipynb
├── scripts/                   # Scripts de Python
│   ├── models/               # Modelos de machine learning
│   │   ├── automlTitanicModel.py
│   │   ├── svmTitanicModel.py
│   │   ├── generativeLanguageModel.py      # Modelo generativo LSTM
│   │   ├── interactiveGenerator.py          # Interfaz interactiva
│   │   ├── quickDemo.py                     # Demostración del modelo
│   │   └── testLstmModel.py                 # Script de verificación
│   └── utils/                # Utilidades (futuro)
└── requirements.txt           # Dependencias del proyecto
```

## Temas Cubiertos

- **Titanic**: Análisis de supervivencia usando SVM y AutoML
- **CNN**: Redes neuronales convolucionales con TensorFlow
- **NLP**: Procesamiento de lenguaje natural
- **VAE**: Autoencoders variacionales para gatos y MNIST
- **Embeddings**: Generación de embeddings y GenAI
- **Visualización**: Herramientas de visualización para CNNs
- **Generativo**: Modelo LSTM para texto religioso (Biblia-Corán)

## Instalación

1. Crear un entorno virtual:
```bash
python -m venv .venv
source .venv/bin/activate  # En macOS/Linux
# o
.venv\Scripts\activate     # En Windows
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Uso

Los notebooks están organizados por tema y pueden ejecutarse independientemente. Los scripts de Python están en la carpeta `scripts/models/` y pueden ejecutarse desde la raíz del proyecto.

### Modelo Generativo de Lenguaje

El modelo generativo LSTM entrenado en el dataset Biblia-Corán permite:

1. **Entrenamiento**: Entrenamiento rápido del modelo LSTM en texto religioso español
2. **Generación**: Predicción automática de 20-100 tokens según el contexto
3. **Interfaz Interactiva**: Generación de texto en tiempo real

**Características del modelo:**
- Arquitectura LSTM de 2 capas con embeddings
- Entrenamiento en 5-10 minutos
- Modelo ligero de ~1MB
- Optimizado para texto religioso en español
- Generación automática de continuaciones

## Convenciones de Nomenclatura

- **Archivos**: camelCase (ej: `titanicLearningNotebook.ipynb`)
- **Carpetas**: camelCase (ej: `catsDogsMiniDataset`)
- **Scripts**: camelCase (ej: `svmTitanicModel.py`)
