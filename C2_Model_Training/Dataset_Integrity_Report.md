# Dataset Integrity Report

## Dataset
- Training: 16,745 PlantVillage images
- Validation: 4,182 PlantVillage images
- Reserved test set: 777 PlantDoc images

## Integrity Checks

1. Class and file ID overlap: 0
2. Exact duplicate files across the actual
   training and validation splits: 0
3. Six identical files were found when scanning
   the validation folder, but none were included
   in the validation CSV.

## Conclusion

No exact file duplicates were detected between
the training and validation images actually used
by the C2 models.

This check does not exclude visually near-identical
images or other potential sources of data leakage.

The PlantDoc test dataset was not used for
C2 training or validation.
