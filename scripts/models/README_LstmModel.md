# Bible-Quran Generative Language Model

Este modelo generativo de lenguaje está entrenado en el dataset combinado de la Biblia y el Corán para predecir y continuar texto religioso y espiritual.

## Características

- **Arquitectura**: Modelo LSTM de 2 capas con embeddings
- **Dataset**: 6000 textos religiosos limpios en español
- **Generación**: Predicción automática de 20-100 tokens según el contexto
- **Velocidad**: Entrenamiento rápido en 5-10 minutos
- **Eficiencia**: Modelo ligero de ~1MB con uso de memoria optimizado

## Arquitectura Técnica

### Modelo LSTM
- **Capas**: 2 capas LSTM + embeddings
- **Embedding**: 128 dimensiones
- **Hidden**: 256 unidades
- **Dropout**: 0.2 para regularización
- **Parámetros**: ~100K parámetros entrenables

### Tokenización
- **Nivel**: Carácter (procesamiento rápido)
- **Vocabulario**: ~25 caracteres únicos
- **Secuencias**: 100 caracteres de entrenamiento
- **Batch**: 32 secuencias por defecto

### Optimización
- **Optimizador**: Adam
- **Learning Rate**: 0.001 por defecto
- **Loss**: CrossEntropyLoss
- **Device**: CPU/GPU automático

## Rendimiento

### Tiempos de Entrenamiento
- **Entrenamiento**: 5-10 minutos en CPU
- **Generación**: Tiempo real
- **Memoria**: ~100MB RAM

### Uso de Recursos
- **Modelo**: ~1MB en disco
- **Memoria**: ~100MB RAM
- **Procesamiento**: CPU eficiente

### Calidad de Generación
- **Continuaciones cortas**: Excelente
- **Continuaciones largas**: Buena
- **Coherencia**: Alta para texto religioso

## Prompts Optimizados

El modelo funciona mejor con prompts relacionados con:
- Espiritualidad y fe
- Amor y misericordia divina
- Bendiciones y sabiduría
- Textos religiosos en español

## Parámetros de Entrenamiento

- **Épocas**: 10-15 para calidad óptima
- **Batch size**: 32-64 para balance velocidad/memoria
- **Learning rate**: 0.001 para convergencia estable
- **Secuencia**: 100 caracteres de contexto

## Parámetros de Generación

- **Temperature**: 0.7-1.0 (controla creatividad)
- **Tokens**: 20-100 (longitud de generación)
- **Longitud óptima**: Determinada automáticamente según el prompt

## Limitaciones

- **Idioma**: Optimizado para español religioso
- **Contexto**: Funciona mejor con temas espirituales
- **Longitud**: Máximo 100 tokens por generación
- **Vocabulario**: Limitado a caracteres vs palabras completas

## Aplicaciones

- Generación de continuaciones de texto religioso
- Experimentación con lenguaje espiritual
- Herramientas educativas de IA
- Investigación en procesamiento de lenguaje religioso

## Tecnología

El modelo utiliza PyTorch con arquitectura LSTM estándar, optimizada para el dominio específico de texto religioso en español, proporcionando un balance entre velocidad de entrenamiento y calidad de generación.
