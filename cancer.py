import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import os
import medmnist
from medmnist import TissueMNIST

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

def evaluate(model, testloader, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for bags, labels in testloader:
            bags = bags.to(device)
            labels = labels.float().to(device)

            outputs = model(bags)
            loss = loss_fn(outputs.squeeze(), labels.squeeze())

            total_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            correct += (predicted == labels.squeeze()).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return total_loss, accuracy

trainset_raw = TissueMNIST(split="train", download=True)
testset_raw = TissueMNIST(split="test", download=True)

def create_bags(dataset, num_bags=200, bag_size=10):
    bags = []
    labels = []
    
    for _ in range(num_bags):
        indices = np.random.randint(0, len(dataset), bag_size)
        bag_patches = [Image.fromarray(np.array(dataset[i][0])) for i in indices]
        bag_label = int(any(dataset[i][1] > 0 for i in indices))
        
        bags.append(bag_patches)
        labels.append(bag_label)
    
    return bags, labels

train_bags, train_labels = create_bags(trainset_raw, num_bags=200)
test_bags, test_labels = create_bags(testset_raw, num_bags=50)

print(f"Training bags: {len(train_bags)}")
print(f"Test bags: {len(test_bags)}")

trainset = MILDataset(train_bags, train_labels, transform=transform)
testset = MILDataset(test_bags, test_labels, transform=transform)

trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

model = MILModel().to(device)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    train_loss, train_acc = train(model, trainloader, optimizer, loss_fn)
    test_loss, test_acc = evaluate(model, testloader, loss_fn)
    
    print(f"Epoch {epoch + 1} | Train Loss: {train_loss:.2f} | Train Acc: {train_acc:.1f}% | Test Acc: {test_acc:.1f}%")

print("Training complete")
torch.save(model.state_dict(), "cancer.pth")
print("Model saved")