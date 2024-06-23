import torch
from torch.utils.data import Dataset
import albumentations
import config
import cv2
import os

class ShopeeDataSet(Dataset):

    def __init__(self, df, transform=None):

        self.image_path = config.train_images_path
        self.df = df
        self.transform = transform
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        observation = self.df.iloc[idx]

        path = os.path.join(self.image_path, observation.image)
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = observation.label_group

        if self.transform:
            augment = self.transform(image=image)
            image = augment['image']
        
        return {
            "image": image,
            "label": torch.tensor(label).long()
        }


def get_transform():

    return albumentations.Compose([

        albumentations.Resize(config.IMG_SIZE, config.IMG_SIZE, always_apply=True),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Rotate(limit=120, p=0.8),
        albumentations.Normalize(mean = config.mean, std = config.std),

        albumentations.pytorch.transform.ToTensorV2(p=1.0)
    ])


def get_test_transform():

    return albumentations.Compose([
        
        albumentations.Resize(config.img_size, config.img_size, always_apply=True),
        albumentations.Normalize(),
        albumentations.pytorch.transforms.ToTensorV2(p=1.0)
    ])