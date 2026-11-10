"""Reusable transforms and DataLoaders, reading cleaned CSV manifests."""
from pathlib import Path
import json
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

def get_transforms(image_size=224):
    train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    evaluate = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return train, evaluate

class PlantDiseaseDataset(Dataset):
    def __init__(self, dataframe, image_root, split, transform):
        self.df = dataframe.reset_index(drop=True)
        self.root = Path(image_root)
        self.split = split
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = self.root / self.split / row['class_name'] / (str(row['file_id']) + Path(row['filename']).suffix.lower())
        with Image.open(path) as image:
            image = self.transform(image.convert('RGB'))
        return image, int(row['label'])

def create_dataloaders(project_root, image_root, batch_size=32, num_workers=0):
    root = Path(project_root)
    with open(root / 'config/class_to_idx.json') as f:
        mapping = json.load(f)
    train_t, eval_t = get_transforms()
    dfs = {split: pd.read_csv(root / 'splits' / f'{split}.csv') for split in ('train', 'valid', 'test')}
    datasets = {
        'train': PlantDiseaseDataset(dfs['train'], image_root, 'train', train_t),
        'valid': PlantDiseaseDataset(dfs['valid'], image_root, 'valid', eval_t),
        'test': PlantDiseaseDataset(dfs['test'], image_root, 'plantdoc_test', eval_t),
    }
    for split, ds in datasets.items():
        if not len(ds):
            raise ValueError(f'{split} is empty')
        invalid = set(ds.df['label']) - set(mapping.values())
        if invalid:
            raise ValueError(f'Invalid labels in {split}: {invalid}')
    loaders = {key: DataLoader(ds, batch_size=batch_size, shuffle=(key=='train'), num_workers=num_workers) for key,ds in datasets.items()}
    counts = dfs['train']['label'].value_counts().reindex(range(len(mapping)), fill_value=0)
    if (counts==0).any():
        raise ValueError('Missing training classes')
    weights = torch.as_tensor(len(dfs['train']) / (len(mapping) * counts.to_numpy()), dtype=torch.float32)
    return {'train':loaders['train'], 'valid':loaders['valid'], 'test':loaders['test'], 'class_weights':weights,'class_to_idx':mapping}
