import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
import medmnist
from medmnist import PathMNIST

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

trainset = PathMNIST(split="train", transform=transform, download=True)
testset = PathMNIST(split="test", transform=transform, download=True)

trainloader = DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
    num_workers=2
)

testloader = DataLoader(
    testset,
    batch_size=64, 
    shuffle=False,
    num_workers=2
)

model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.linear(512, 9)
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

for epoch in range(5):
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in trainloader:
        labels = labels.squeeze().long()
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Epoch {epoch + 1} | Loss: {total_loss:.2f} | Accuracy: {accuracy:.1f}%")

    print("Training complete")

    torch.save(model.state_dict(), "medical.pth")
    print("Model saved")

    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.1f}%")