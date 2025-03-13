# predict.py

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pickle

# -----------------------------
# 1. Define the Multi-Output Model
# -----------------------------

class MultiOutputModel(nn.Module):
    def __init__(self, num_subtypes):
        super(MultiOutputModel, self).__init__()
        # Use the same architecture as in the training script
        self.base_model = models.resnet18(pretrained=False)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()

        # Output layers
        self.fc_binary = nn.Linear(num_features, 1)
        self.fc_subtype = nn.Linear(num_features, num_subtypes)

    def forward(self, x):
        x = self.base_model(x)
        binary_output = torch.sigmoid(self.fc_binary(x))
        subtype_output = self.fc_subtype(x)
        return binary_output, subtype_output

# -----------------------------
# 2. Load the Model and Class Indices
# -----------------------------

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load class_to_idx mapping
with open('class_to_idx.pkl', 'rb') as f:
    class_to_idx = pickle.load(f)

num_subtypes = len(class_to_idx)

# Initialize the model
model = MultiOutputModel(num_subtypes=num_subtypes)
model.load_state_dict(torch.load('multi_output_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Create idx_to_class mapping
idx_to_class = {v: k for k, v in class_to_idx.items()}

# -----------------------------
# 3. Define the Prediction Function
# -----------------------------

def predict_image(model, image_path, transform):
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file '{image_path}' not found.")
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output_binary, output_subtype = model(image)
        output_binary = output_binary.item()
        predicted_binary = 'Benign' if output_binary < 0.5 else 'Malignant'
        confidence_binary = (1 - output_binary) * 100 if output_binary < 0.5 else output_binary * 100

        _, predicted_subtype_idx = torch.max(output_subtype, 1)
        predicted_subtype_idx = predicted_subtype_idx.item()
        predicted_subtype = idx_to_class[predicted_subtype_idx]

    print(f"Prediction:")
    print(f" - Benign/Malignant: {predicted_binary} ({confidence_binary:.2f}% confidence)")
    print(f" - Subtype: {predicted_subtype}")

    return predicted_binary, predicted_subtype

# -----------------------------
# 4. Define Transforms and Predict
# -----------------------------

# Use the same transforms as validation
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

if __name__ == '__main__':
    # Path to the image you want to predict
    image_path = r"C:\Users\eyita\Documents\tt.png"  # Replace with your image path

    # Predict the image
    predict_image(model, image_path, val_transforms)
