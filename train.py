import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

trainset = torchvision.datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

print("Dataset loaded")

trainloader = DataLoader(
    trainset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(2):
    total_loss = 0

    print(f"Starting epoch {epoch + 1}")

    for images, labels in trainloader:
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1} Loss: {total_loss:.2f}")
    
print("Training complete")