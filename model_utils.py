import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import streamlit as st
import numpy as np
import cv2
import os
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Load pre-trained ResNet18 with custom classifier for 2 classes (Real vs Fake)
def load_model(path):
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # Output 2 classes
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Predict a single image and return label and confidence percentage
def predict_image(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)  # Add batch dim
    output = model(img_tensor)
    probs = F.softmax(output, dim=1).detach().numpy().flatten()
    label = "Authentic Content" if probs[0] > probs[1] else "AI-Generated Content"
    confidence = float(probs.max()) * 100
    return label, confidence

# Dummy placeholder for video prediction (simulate output)
def predict_video(video_path, model):
    # You can replace this with actual video frame prediction logic
    output = torch.tensor([[0.4, 0.6]])
    probs = F.softmax(output, dim=1).detach().numpy().flatten()
    label = "Authentic Content" if probs[0] > probs[1] else "AI-Generated Content"
    confidence = float(probs.max()) * 100
    return label, confidence

# Plot confidence gauge using Matplotlib and save it (replaces unreliable Plotly version)
def confidence_gauge(confidence, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlim(-1, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background arc
    arc = patches.Wedge((0, 0), 1, 0, 180, facecolor='lightgray')
    ax.add_patch(arc)

    # Colored sections
    colors = ['red', 'orange', 'yellow', 'lime', 'green']
    angles = [0, 36, 72, 108, 144, 180]
    for i in range(5):
        wedge = patches.Wedge((0, 0), 1, angles[i], angles[i+1], facecolor=colors[i])
        ax.add_patch(wedge)

    # Needle
    angle = (confidence / 100) * 180
    x = 0.8 * np.cos(np.radians(180 - angle))
    y = 0.8 * np.sin(np.radians(180 - angle))
    ax.plot([0, x], [0, y], color='black', linewidth=3)

    # Text
    ax.text(0, -0.2, f"Confidence: {confidence:.1f}%", ha="center", fontsize=14, fontweight="bold")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f"✅ Gauge saved to {save_path}")
        plt.close()
    else:
        st.pyplot(fig)

# Generate Grad-CAM heatmap for an input image and return PIL image with heatmap overlay
def generate_heatmap(model, image, save_path=None):
    model.eval()
    gradients = []
    activations = []

    # Forward hook to save activations
    def forward_hook(module, input, output):
        activations.append(output)

    # Backward hook to save gradients
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks on the last convolutional layer
    target_layer = model.layer4[-1]
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_full_backward_hook(backward_hook)  # Updated for PyTorch >=1.8

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0)
    input_tensor.requires_grad = True

    # Forward pass
    output = model(input_tensor)
    class_idx = output.argmax().item()

    # Zero grads and backward pass on target class score
    model.zero_grad()
    output[0, class_idx].backward()

    # Get gradients and activations from hooks
    grads = gradients[0]       # shape: [batch_size, channels, height, width]
    acts = activations[0]      # shape: [batch_size, channels, height, width]

    # Global average pooling of gradients over spatial dimensions
    pooled_grads = torch.mean(grads, dim=[0, 2, 3])

    # Weight the activations by corresponding pooled gradients
    for i in range(acts.shape[1]):
        acts[:, i, :, :] *= pooled_grads[i]

    # Create heatmap by averaging weighted activations across channels
    heatmap = torch.mean(acts, dim=1).squeeze().detach().numpy()

    # Normalize heatmap to [0,1]
    heatmap = np.maximum(heatmap, 0)
    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)

    # Convert image and heatmap to appropriate sizes and types
    image_np = np.array(image.resize((224, 224)))  # PIL to numpy
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Overlay heatmap on image (with transparency)
    superimposed = cv2.addWeighted(image_np, 0.6, heatmap_color, 0.4, 0)

    # Convert back to PIL image
    final_image = Image.fromarray(superimposed)

    # Save if save path provided
    if save_path:
        final_image.save(save_path)

    return final_image
