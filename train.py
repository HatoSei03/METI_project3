import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Define the CNN architecture with batch normalization
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Data transformation and augmentation
transform = transforms.Compose([
    transforms.RandomRotation(10),  # Random rotation up to 10 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Slight translation
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

# Load the MNIST data
full_dataset = torchvision.datasets.MNIST(root='D:/exam/project3/METI_project3/data', train=True, download=True, transform=transform)
train_size = int(0.9 * len(full_dataset))  # 90% for training
val_size = len(full_dataset) - train_size  # 10% for validation
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Initialize the CNN model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Reduce LR by 0.1 every 5 epochs

# Training loop with validation
epochs = 20
best_val_accuracy = 0.0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # Validation phase
    model.eval()
    correct_val = 0
    total_val = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            output = model(images)
            loss = criterion(output, labels)
            val_loss += loss.item()
            _, predicted = torch.max(output, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    avg_train_loss = running_loss / len(train_loader)
    train_accuracy = (correct_train / total_train) * 100
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = (correct_val / total_val) * 100

    scheduler.step()  # Update learning rate

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # Save the best model
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'METI_project3/mnist_model.pth')
        print("Saved best model with validation accuracy: {:.2f}%".format(best_val_accuracy))

print("Model training complete. Best model saved to 'mnist_model.pth'")