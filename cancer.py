import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class MILDataset(Dataset):
    def __init__(self, bags, labels, transform=None):
        self.bags = bags
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        bag = self.bags[idx]
        label = self.labels[idx]

        patches = []
        for patch in bag:
            if self.transform:
                patch = self.transform(patch)
            patches.append(patch)

        return torch.stack(patches), label
    