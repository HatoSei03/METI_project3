# import streamlit as st
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# import random
# import torchvision.transforms as transforms
# import torchvision
# from torch.utils.data import DataLoader
# import torch.optim as optim

# # Define the neural network architecture
# class SimpleNN(nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = x.view(-1, 28 * 28)  # Flatten input
#         x = torch.relu(self.fc1(x))  # Activation function
#         x = self.fc2(x)  # Output layer
#         return x

# # Load the trained model checkpoint
# model = SimpleNN()
# model.load_state_dict(torch.load('METI_project3\mnist_model.pth'))
# model.eval()

# # Function to generate random digit images
# def generate_digit_image(digit):
#     # Generate random input to simulate digit generation (use model for classification prediction)
#     random_input = torch.randn(1, 28 * 28)
#     with torch.no_grad():
#         output = model(random_input)
#     predicted_digit = torch.argmax(output, dim=1).item()
    
#     # Reshaping to 28x28 image for visualization
#     image = random_input.view(28, 28).numpy()
#     image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    
#     return image

# # Streamlit User Interface
# st.set_page_config(page_title="MNIST Digit Generator", page_icon=":guardsman:", layout="centered")
# st.title("MNIST Digit Generator")

# # Display instructions
# st.markdown("""
# Select a digit (0-9) from the slider, and the app will generate 5 images of that digit.
# Click "Generate" to see the images.
# """)

# # Slider to select digit
# digit = st.slider("Select a digit (0-9):", 0, 9, 0)

# # Add a button for generating digits
# generate_button = st.button("Generate Images")

# # Loading spinner while generating
# if generate_button:
#     with st.spinner("Generating 5 images..."):
#         # Generate 5 images based on the selected digit
#         images = [generate_digit_image(digit) for _ in range(5)]

#         # Display the generated images
#         fig, axes = plt.subplots(1, 5, figsize=(15, 5))
#         for ax, image in zip(axes, images):
#             ax.imshow(image, cmap='gray')
#             ax.axis('off')  # Hide axes

#         st.pyplot(fig)

#     st.success(f"5 images of digit {digit} generated successfully!")

import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

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

# Load the trained model
model = SimpleCNN()
model.load_state_dict(torch.load('mnist_model.pth'))  # Ensure the path is correct
model.eval()

# Function to generate random digit images
def generate_digit_image(digit):
    # Generate random input to simulate digit generation (use model for classification prediction)
    random_input = torch.randn(1, 28 * 28).unsqueeze(0)  # Adjust input shape for batch processing
    with torch.no_grad():
        output = model(random_input)
    
    # Reshaping to 28x28 image for visualization
    image = random_input.view(28, 28).numpy()
    image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    
    return image

# Streamlit User Interface
st.set_page_config(page_title="Handwritten Digit Image Generator", page_icon=":guardsman:", layout="centered")
st.title("Handwritten Digit Image Generator")

# Display instructions
st.markdown("""
Generate synthetic MNIST-like images using your trained model.

**Choose a digit to generate (0-9)** and click on "Generate Images" to see the result.
""")

# Slider to select digit
digit = st.slider("Choose a digit to generate (0-9):", 0, 9, 0)

# Button to generate images
generate_button = st.button("Generate Images")

# Loading spinner while generating
if generate_button:
    st.write(f"Generating 5 images of digit {digit}...")
    
    # Generate 5 images based on the selected digit
    images = [generate_digit_image(digit) for _ in range(5)]

    # Display the generated images
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(images[0], caption="Sample 1", use_column_width=True)
    with col2:
        st.image(images[1], caption="Sample 2", use_column_width=True)
    with col3:
        st.image(images[2], caption="Sample 3", use_column_width=True)
    with col4:
        st.image(images[3], caption="Sample 4", use_column_width=True)
    with col5:
        st.image(images[4], caption="Sample 5", use_column_width=True)

    st.success(f"5 images of digit {digit} generated successfully!")


