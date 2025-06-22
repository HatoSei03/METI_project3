import torch
import torch.nn as nn
import numpy as np
import random
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
import torch.optim as optim

# Define the CNN architecture suitable for grayscale images (1 channel)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 1 channel for grayscale
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjusted for smaller input size (28x28 -> 7x7 after pooling)
        self.fc2 = nn.Linear(128, 10)  # Output layer with 10 classes (for MNIST digits)
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor to feed into the fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = self.fc2(x)
        return x

# Data transformation and normalization
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor (shape: [1, 28, 28] for MNIST)
    transforms.Normalize((0.5,), (0.5,))  # Normalize the grayscale images (mean=0.5, std=0.5)
])

# Load the MNIST data directly using torchvision.datasets.MNIST
train_data = torchvision.datasets.MNIST(root='D:/exam/project3/METI_project3/data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Initialize the CNN model, loss function, and optimizer
model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0
    correct_predictions = 0
    total_predictions = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        # Forward pass
        output = model(images)
        loss = criterion(output, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(output, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = (correct_predictions / total_predictions) * 100
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

# Save the trained model
torch.save(model.state_dict(), 'mnist_model.pth')
print("Model training complete and saved to 'mnist_model.pth'")
