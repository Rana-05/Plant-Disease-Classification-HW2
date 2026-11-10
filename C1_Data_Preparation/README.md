# Plant Disease Classification HW2 — C1 handover

**Status:** source notebook was reviewed and reused. Code is packaged; **dataset regeneration is still required**, because the original download and generated manifests were missing after a Colab reset. This package does not contain the images or cleaned CSVs.

## Included
- `C1_original_notebook.ipynb`: original record of implementation and prior successful tests.
- `C1_Handover_Setup.ipynb`: ordered setup and smoke-test notebook.
- `prepare_from_shared_drive.py`: lists original shared Drive dataset, downloads images, checks image validity and exact SHA-256 duplicates, saves clean CSVs and class weights.
- `c1_pipeline.py`: reusable PyTorch train/valid/test loaders and on-the-fly augmentations.
- `models.py`: pretrained FasterNet, EfficientViT, GhostNet with 9-class heads.
- `config/class_to_idx.json`: standardized nine-class mapping.

## Dataset and setup
- Original shared Google Drive folder: https://drive.google.com/drive/u/0/folders/1rkAM3SPGkMbStjyfk98fxIFOVB3EFZSK
- Selected crops: Apple, Grape, Tomato. Exclude Corn.
- Use source `Plant_Disease_Detection_Dataset/train` and `/valid`, **not** its `/test`, for laboratory training and validation.
- Use `PlantDoc_field` only for final testing, not model selection or training.
- In Google Colab open `C1_Handover_Setup.ipynb` and run in order. Download `C1_GitHub_Handover.zip` from the GitHub repository and upload it when the notebook prompts. Select a writable Drive folder for the prepared images. The first setup requires downloading the original files **once**. It cannot be skipped if no other accessible image copy exists.
- C2 must have Drive permissions to read the original shared folder and the prepared data folder. GitHub does **not** contain datasets.

## Preprocessing
Training: random resized crop 224, random horizontal flip, ±15° rotation, color jitter, tensor, ImageNet normalization. Validation/test: resize 224, tensor, ImageNet normalization. Transforms occur at load time; no need to store preprocessed images separately.

## Data counts and cleaning
Original notebook recorded 16,766 train / 4,190 valid / 778 field images. It recorded cleaned 16,745 train / 4,182 valid / 778 field samples and no exact cross-split hashes. **These counts are historical outputs, not guaranteed for reconstructed manifests.** The setup script re-creates manifests from currently accessible source files, downloads/validates each image, checks duplicate hashes, and regenerates weights. A cross-split byte-hash check does not detect near-duplicate or similarly augmented photos.

## For C2
After the preparation notebook finishes, import `create_dataloaders` from `c1_pipeline` and `initialize_model` from `models`. Example:

```python
from c1_pipeline import create_dataloaders
from models import initialize_model
loaders = create_dataloaders(project_root=PROJECT_ROOT, image_root=IMAGE_ROOT)
model = initialize_model('FasterNet', num_classes=len(loaders['class_to_idx']))
train_loader, valid_loader = loaders['train'], loaders['valid']
class_weights = loaders['class_weights']
```

`test` is for **C3** only. `torch.nn.CrossEntropyLoss(weight=class_weights.to(device))` is one way C2 may use weights. C2 is responsible for training and saving checkpoints.

## Persistence / source control
- Colab `/content/` is temporary; everything needed after a runtime reset must be stored in My Drive or another persistent store. Files on Drive may take time to sync; confirm their presence in the Drive web UI before disconnecting.
- Upload these source files and notebooks to GitHub; **do not commit** `prepared_data/`, model checkpoints, or private credentials. `splits/*.csv` may be committed after regeneration (they contain file IDs but no image bytes). Review sharing permissions.
