# Plant Disease Classification — Homework II

## Project Overview

This project evaluates three lightweight deep learning models
for plant disease classification across Apple, Grape, and Tomato
crops.

The project consists of three components:

- C1: Data Preparation and Model Setup
- C2: Model Training
- C3: Classification Evaluation and Explainability

## Models

1. FasterNet
2. EfficientViT
3. GhostNet

## Dataset

Training and validation use PlantVillage-style laboratory images.

Final classification testing uses 777 PlantDoc field images
across 9 classes.

The PlantDoc test set is reserved for evaluation and is not
used for model training.

## Project Structure

C1_Data_Preparation/
    Complete prepared datasets, configurations, splits,
    and preserved GitHub source files.

C2_Model_Training/
    Training notebooks, checkpoints, logs, and graphs.

C3_Evaluation_Explainability/
    Classification testing, confusion matrices, Grad-CAM,
    Insertion & Deletion, and final results.

Shared_Config/
    Shared configuration copies and provenance manifest.

Report_Resources/
    Selected tables and figures for report preparation.

Original_Repositories/
    Preserved C1 and C2 Git repositories, including Git history.

Verification/
    File integrity and repository verification reports.

## Classification Evaluation

All three trained models were tested on 777 PlantDoc images.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| FasterNet | 0.3745 | 0.5094 | 0.3837 | 0.3609 |
| EfficientViT | 0.3436 | 0.5535 | 0.3416 | 0.3133 |
| GhostNet | 0.3449 | 0.5447 | 0.3547 | 0.3308 |

## Grad-CAM

Grad-CAM visualizations were generated for 18 selected
PlantDoc images, with all three models evaluated on
the same images.

Total visualizations: 54.

## Insertion and Deletion

Insertion and Deletion were evaluated on 90 PlantDoc images,
with 10 images selected per class.

Each model was evaluated on the same 90 images.

Total evaluations: 270.

| Model | Insertion AUC | Deletion AUC |
|---|---:|---:|
| FasterNet | 0.6117 | 0.2542 |
| EfficientViT | 0.5808 | 0.3436 |
| GhostNet | 0.6527 | 0.2881 |

Higher Insertion AUC and lower Deletion AUC are preferred.

## Model Comparison

FasterNet achieved the highest PlantDoc classification
accuracy, macro recall, and macro F1.

GhostNet achieved the highest mean Insertion AUC.

FasterNet achieved the lowest mean Deletion AUC.

These results indicate different strengths in classification
performance and explanation faithfulness.

## Reproducibility

The original notebooks and model checkpoints are preserved.

The code was originally developed using separate Google Drive
folders. Some notebook paths may require updating when
running from the unified project directory.

Copying files does not require retraining, but reproducing
the complete pipeline in a new environment may require
path adjustments and dependency installation.

## Data Integrity

The original C1, C2, and C3 Google Drive files were copied
and verified using SHA-256.

22,158 files matched their original sources during the
initial verification.

Additional C1 GitHub source files were preserved separately.

See Verification/ for integrity reports.

## Original GitHub Repositories

C1:
https://github.com/Jawaher-maa/Plant-Disease-Classification-HW2-6..git

C2:
https://github.com/joryabadgaish/Plant-Disease-Classification-C2-Model-Trainin

## Notes for the Report Team

- Use PlantDoc results for final field-image performance.
- Do not confuse C2 validation accuracy with C3 test accuracy.
- Report macro recall alongside accuracy, precision, and F1.
- Use the Grad-CAM figures for qualitative explainability.
- Use Insertion and Deletion AUC for quantitative
  explanation faithfulness.
- The full dataset, checkpoints, and detailed outputs
  are preserved in their respective component folders.
