# backend/utils.py
import tensorflow as tf
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def preprocess_image(filepath, target_size=(224, 224)):
    """
    Preprocess image for TensorFlow model prediction
    
    Args:
    - filepath (str): Path to the image file
    - target_size (tuple): Desired image size for model input
    
    Returns:
    - Preprocessed image tensor ready for model prediction
    """
    # Open the image
    img = Image.open(filepath)
    
    # Resize the image
    img = img.resize(target_size)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Normalize the image (assuming model expects values between 0 and 1)
    img_array = img_array / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def preprocess_efficientnet_image(filepath, variety_index, target_size=(224, 224)):
    """
    Preprocess image for EfficientNet model with variety input
    
    Args:
    - filepath (str): Path to the image file
    - variety_index (int): Index of the variety
    - target_size (tuple): Desired image size for model input
    
    Returns:
    - Preprocessed image and variety tensor for PyTorch model
    """
    # Open and preprocess image
    img = Image.open(filepath)
    img = img.resize(target_size)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Transform image
    img_tensor = transform(img)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    # Convert variety index to tensor
    variety_tensor = torch.tensor([variety_index], dtype=torch.long)
    
    return img_tensor, variety_tensor   