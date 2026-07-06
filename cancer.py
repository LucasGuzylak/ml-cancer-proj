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
    
class MILModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        resnet = models.resnet18(weights="IMAGENET1K_V1")
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
    
    def forward(self, bag):
        bag = bag.squeeze(0)
        
        features = self.feature_extractor(bag)
        features = features.view(features.size(0), -1)
        
        patch_scores = self.classifier(features)
        
        bag_score = patch_scores.max(dim=0)[0]
        
        return bag_score
    
def train(model, trainloader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for bags, labels in trainloader:
        bags = bags.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(bags)
        loss = loss_fn(outputs.squeeze(), labels.squeeze())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = (outputs.squeeze() > 0.5).float()
        correct += (predicted == labels.squeeze()).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    return total_loss, accuracy