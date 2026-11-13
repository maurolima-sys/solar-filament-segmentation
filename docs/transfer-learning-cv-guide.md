# Guia de Estudo — Transfer Learning for Computer Vision (Kaggle)

## 1. Introdução

A página **Transfer Learning for CV Guide**, disponibilizada pelo Kaggle, apresenta recursos e notebooks para aprender a utilizar *Transfer Learning* em diferentes tarefas de Computer Vision (CV).

A Computer Vision é uma área científica dedicada ao desenvolvimento de modelos capazes de interpretar imagens e vídeos.

O *Transfer Learning* consiste em utilizar um modelo previamente treinado para uma determinada finalidade e adaptá-lo a uma tarefa diferente.

A principal vantagem desta abordagem é evitar o treino de um modelo desde o início (*training from scratch*), que geralmente exige mais tempo de execução e maior capacidade computacional.

## 2. Como utilizar os tutoriais do Kaggle

O Kaggle recomenda a utilização dos notebooks disponibilizados na página como ponto de partida para a aprendizagem prática.

### Procedimento recomendado

1. Abrir o tutorial pretendido.
2. Clicar em **Copy & Edit**, no canto superior direito.
3. Criar uma cópia privada do notebook.
4. Executar e analisar o código.
5. Estudar a forma como o modelo pré-treinado é utilizado na tarefa apresentada.
6. Caso o notebook seja útil, o Kaggle recomenda deixar um voto positivo (*upvote*).

## 3. Categorias de Computer Vision e notebooks disponíveis

### 3.1. Multiclass Classification

| Notebook | Modelo | Aplicação |
|---|---|---|
| Cats or Dogs - using CNN with Transfer Learning | ResNet pré-treinada | Classificação de gatos e cães |
| Brain Tumor MRI Classification: TensorFlow CNN | EfficientNet pré-treinada | Deteção de tumores cerebrais em MRI |
| EfficientNetB5 with Keras (APTOS 2019) | EfficientNet pré-treinada | Retinopatia diabética |

### 3.2. Multilabel Classification

- **pretrained ResNet34 with RGBY (0.460 public LB)** — iafoss — Human Protein Atlas Image Classification.

### 3.3. Object Detection

| Notebook | Modelo | Aplicação |
|---|---|---|
| End to End Object Detection with Transformers: DETR | DETR | Localização de cabeças de trigo |
| Airbus Mask-RCNN and COCO transfer learning | Mask R-CNN | Localização de navios |

### 3.4. Segmentation

| Notebook | Modelo | Aplicação |
|---|---|---|
| Sartorius - Starter Torch Mask R-CNN [LB=0.273] | Mask R-CNN pré-treinada | Segmentação de células neuronais |
| Mask-RCNN and Medical Transfer Learning SIIM-ACR | Mask R-CNN | Segmentação de pneumotórax |

### 3.5. Emotion Recognition

- **Facial Emotion Recognition using Transfer Learning** — gauravsharma99 — MobileNet.
- **Emotion Detection** — odins0n — DenseNet.

### 3.6. Pose Estimation

- **CenterNet Baseline** — hocop1 — localização de automóveis.

### 3.7. Anomaly Detection

- **Video Anomaly Detection** — odins0n — DenseNet — atividades anómalas em vídeo.

## 4. Aplicação ao projeto Solar Filament Segmentation

A categoria mais diretamente relacionada com o projeto é **Segmentation** — o objetivo é produzir máscaras ao nível dos píxeis para filamentos solares em imagens H-alpha.

### Relação com a arquitetura do projeto

O projeto usa U-Net com encoder ResNet34 (`encoder_weights="imagenet"`). O Transfer Learning já está em uso: o encoder é inicializado com pesos ImageNet e o fine-tuning ocorre durante o treino no MAGFiLO.

O notebook **pretrained ResNet34 with RGBY** é relevante para estudar a utilização de ResNet34, embora a tarefa seja classificação multilabel e não segmentação.

**Importante:** os notebooks do Kaggle são exemplos de outras aplicações. A página não demonstra que Mask R-CNN seja superior à U-Net nem que deva substituir a arquitectura actual.

## 5. Prioridade de estudo para o projeto

| Prioridade | Categoria | Motivo |
|---|---|---|
| 1 | Segmentation | Tarefa directamente relacionada |
| 2 | Multilabel Classification — ResNet34 | Estudo do encoder usado no projecto |
| 3 | Object Detection | Arquitecturas de localização e segmentação |
| 4 | Multiclass Classification | Fundamentos de Transfer Learning |
| 5 | Restantes | Conhecimento complementar |

## 6. Distinção de conceitos

- **Classification**: classe associada a uma imagem.
- **Object Detection**: localização de objectos via bounding boxes.
- **Segmentation**: região de cada objecto ao nível dos píxeis — **objectivo do projecto**.

## 7. Conclusão

Conteúdos mais relevantes para o Solar Filament Segmentation:

1. Segmentação com Mask R-CNN (referência arquitectural).
2. Utilização de ResNet34 pré-treinada.
3. Transfer Learning com redes convolucionais.
4. Adaptação de modelos a diferentes domínios de imagens.

**Recomendação:** leitura em paralelo com a execução da versão 6. Não alterar o pipeline enquanto as verificações de reprodutibilidade estiverem pendentes.

**Fonte:** Kaggle — Transfer Learning for CV Guide
https://www.kaggle.com/learn/transfer-learning-for-cv
