# C3 — Classification Evaluation & Explainability

## Project Overview
This module evaluates three deep learning models for plant disease
classification on real-field PlantDoc images and investigates their
predictions using Grad-CAM, Insertion, and Deletion.

## Models
- FasterNet (fasternet_t0.in1k)
- EfficientViT (efficientvit_b0.r224_in1k)
- GhostNet (ghostnet_100.in1k)

## Dataset
- Test dataset: PlantDoc (777 images)
- Classes: 9
- Crops: Apple, Grape, Tomato
- PlantDoc is reserved exclusively for final testing.

## Evaluation Methods
1. Classification: Accuracy, Macro Precision, Macro Recall,
   Macro F1, confusion matrices, and per-class reports.
2. Grad-CAM: 18 selected PlantDoc images (2 per class),
   evaluated across all three models.
3. Insertion & Deletion: 90 randomly selected images
   (10 per class), using the same samples for all models.
   Evaluation uses 20 perturbation steps, a black RGB baseline,
   and the original predicted class as the fixed target.
   AUC is computed using trapezoidal integration.

## Final Results

| Model | Accuracy | Macro Recall | Macro F1 | Insertion AUC | Deletion AUC |
|---|---:|---:|---:|---:|---:|
| FasterNet | 0.3745 | 0.3837 | 0.3609 | 0.6117 | 0.2542 |
| EfficientViT | 0.3436 | 0.3416 | 0.3133 | 0.5808 | 0.3436 |
| GhostNet | 0.3449 | 0.3547 | 0.3308 | 0.6527 | 0.2881 |

Higher Insertion AUC and lower Deletion AUC are preferable
under the same evaluation protocol.

## Directory Structure

- code/ : Colab evaluation notebook
- classification/ : Predictions, classification reports,
  confusion matrices, and model comparison
- gradcam/ : Selected samples, heatmaps, and visualizations
- insertion_deletion/ : Evaluation samples, AUC results,
  individual curves, and comparison figures
- final/ : Combined model comparison CSV

## How to Run
1. Open `code/C3_Evaluation_Explainability.ipynb`
   in Google Colab.
2. Enable a GPU runtime.
3. Mount Google Drive.
4. Ensure C1 prepared test data and C2 trained checkpoints
   are available in the expected Drive directories.
5. Run the notebook cells sequentially.

## Dependencies
Python, PyTorch, torchvision, timm, NumPy, pandas,
Pillow, scikit-learn, and Matplotlib.

## Notes
- No PlantDoc images were used for training or tuning.
- The same preprocessing and class mapping are used
  across all three models.
- Classification and explanation faithfulness measure
  different aspects of model behavior.
- Grad-CAM highlights influential image regions but
  does not establish that the model identified genuine
  disease symptoms.
- Insertion and Deletion results depend on the masking
  baseline and perturbation strategy.
