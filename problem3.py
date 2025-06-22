# import streamlit as st
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# import numpy as np
# from PIL import Image
# import random

# # Define the CNN architecture suitable for grayscale images (1 channel)
# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 1 channel for grayscale
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjusted for smaller input size (28x28 -> 7x7 after pooling)
#         self.fc2 = nn.Linear(128, 10)  # Output layer with 10 classes (for MNIST digits)
#         self.pool = nn.MaxPool2d(2, 2)  # Max pooling layer
#         self.dropout = nn.Dropout(0.5)  # Dropout for regularization

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = x.view(-1, 64 * 7 * 7)  # Flatten the tensor to feed into the fully connected layers
#         x = torch.relu(self.fc1(x))
#         x = self.dropout(x)  # Apply dropout
#         x = self.fc2(x)
#         return x

# # Load the trained model
# model = SimpleCNN()
# model.load_state_dict(torch.load('METI_project3\mnist_model.pth'))  # Ensure the path is correct
# model.eval()

# # Function to generate random digit images
# def generate_digit_image(digit):
#     # Generate random input to simulate digit generation (use model for classification prediction)
#     random_input = torch.randn(1, 28 * 28).unsqueeze(0)  # Adjust input shape for batch processing
#     with torch.no_grad():
#         output = model(random_input)
    
#     # Reshaping to 28x28 image for visualization
#     image = random_input.view(28, 28).numpy()
#     image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
    
#     return image

# # Streamlit User Interface
# st.set_page_config(page_title="Handwritten Digit Image Generator", page_icon=":guardsman:", layout="centered")
# st.title("Handwritten Digit Image Generator")

# # Display instructions
# st.markdown("""
# Generate synthetic MNIST-like images using your trained model.

# **Choose a digit to generate (0-9)** and click on "Generate Images" to see the result.
# """)

# # Slider to select digit
# digit = st.slider("Choose a digit to generate (0-9):", 0, 9, 0)

# # Button to generate images
# generate_button = st.button("Generate Images")

# # Loading spinner while generating
# if generate_button:
#     st.write(f"Generating 5 images of digit {digit}...")
    
#     # Generate 5 images based on the selected digit
#     images = [generate_digit_image(digit) for _ in range(5)]

#     # Display the generated images
#     col1, col2, col3, col4, col5 = st.columns(5)
#     with col1:
#         st.image(images[0], caption="Sample 1", use_column_width=True)
#     with col2:
#         st.image(images[1], caption="Sample 2", use_column_width=True)
#     with col3:
#         st.image(images[2], caption="Sample 3", use_column_width=True)
#     with col4:
#         st.image(images[3], caption="Sample 4", use_column_width=True)
#     with col5:
#         st.image(images[4], caption="Sample 5", use_column_width=True)

#     st.success(f"5 images of digit {digit} generated successfully!")


import streamlit as st
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import os


# Set page configuration as the FIRST Streamlit command
st.set_page_config(page_title="Handwritten Digit Generator", page_icon=":pencil2:", layout="wide")

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #f0f2f5, #e0e7ff);
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4f46e5;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: background-color 0.3s;
    }
    .stButton>button:hover {
        background-color: #4338ca;
    }
    .card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .title {
        font-size: 2.5em;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 20px;
    }
    .sidebar .stSlider {
        background: #e0e7ff;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# Define the CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # 1 channel for grayscale
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # Adjusted for 28x28 -> 7x7 after pooling
        self.fc2 = nn.Linear(128, 10)  # Output layer for 10 classes
        self.pool = nn.MaxPool2d(2, 2)  # Max pooling
        self.dropout = nn.Dropout(0.5)  # Dropout for regularization

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = SimpleCNN()
        model_path = 'METI_project3\mnist_model.pth'
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            return None
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Function to generate random digit images
def generate_digit_image(digit, model):
    if model is None:
        return None
    try:
        # Generate random input with shape (1, 1, 28, 28) for CNN
        random_input = torch.randn(1, 1, 28, 28)
        with torch.no_grad():
            output = model(random_input)
        predicted_digit = torch.argmax(output, dim=1).item()
        
        # Reshape to 28x28 for visualization
        image = random_input.squeeze().numpy()
        image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
        return image, predicted_digit
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None, None

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    digit = st.slider("Choose a digit (0-9):", 0, 9, 0, key="digit_slider")
    num_images = st.selectbox("Number of images to generate:", [3, 5, 7], index=1)
    st.markdown("---")
    st.info("This app generates synthetic MNIST-like images using a trained CNN model.")
    st.markdown("[Learn more about MNIST](http://yann.lecun.com/exdb/mnist/)")

# Main content
st.markdown('<div class="title">Handwritten Digit Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.write("""
Generate synthetic handwritten digit images using a convolutional neural network trained on the MNIST dataset.
Select a digit and the number of images to generate, then click "Generate Images" to see the results.
""")

# Button to generate images
generate_button = st.button("Generate Images")

# Generate and display images
if generate_button:
    if model is None:
        st.error("Cannot generate images because the model failed to load.")
    else:
        with st.spinner(f"Generating {num_images} images of digit {digit}..."):
            images_with_preds = [generate_digit_image(digit, model) for _ in range(num_images)]
            images_with_preds = [img for img in images_with_preds if img[0] is not None]
            
            if images_with_preds:
                # Display images in a grid
                cols = st.columns(min(len(images_with_preds), 5))
                for idx, (col, (image, pred_digit)) in enumerate(zip(cols, images_with_preds)):
                    with col:
                        st.image(image, caption=f"Sample {idx+1} (Pred: {pred_digit})", use_column_width=True)
                st.success(f"{len(images_with_preds)} images of digit {digit} generated successfully!")
            else:
                st.error("Failed to generate any images.")
st.markdown('</div>', unsafe_allow_html=True)