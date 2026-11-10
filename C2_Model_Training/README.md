# C2 - Model Training

## Overview
This folder contains the training results for three
plant disease classification models:
FasterNet, EfficientViT, and GhostNet.

## Dataset
- Training: 16,745 PlantVillage images
- Validation: 4,182 PlantVillage images
- Classes: 9
- PlantDoc test images (777) are reserved for C3.

## Training Configuration
- Framework: PyTorch
- Pretrained models: timm
- Optimizer: AdamW
- Learning rate: 0.0001
- Weight decay: 0.0001
- Batch size: 32
- Maximum epochs: 30
- Early stopping patience: 7
- Scheduler: ReduceLROnPlateau
- Loss function: CrossEntropyLoss

## Results

| Model | Epochs | Best Epoch | Validation Accuracy | Validation Loss |
|---|---:|---:|---:|---:|
| FasterNet | 20 | 12 | 99.90% | 0.0049 |
| EfficientViT | 23 | 16 | 99.95% | 0.0035 |
| GhostNet | 18 | 11 | 99.90% | 0.0048 |

Best checkpoints were selected using the lowest
validation loss.

## Folder Structure
- checkpoints/: Best and latest model checkpoints
- graphs/: Accuracy and loss plots
- code/: Colab training notebook
- C2_Model_Comparison.csv: Model comparison results

## Notes
- Training and validation used PlantVillage images.
- PlantDoc test data were not used for C2 training.
- Field-image evaluation will be performed in C3.
- Best checkpoints should be used for model evaluation.
