
# multi_output_classification.py

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import matplotlib.pyplot as plt
import pickle

# Custom Dataset class for loading tumor images and their associated labels
class TumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir  # Path to the dataset root directory
        self.transform = transform  # Image transformations
        self.image_paths = []  # List to store image file paths
        self.binary_labels = []  # Binary classification labels (0 or 1)
        self.subtype_labels = []  # Subtype classification labels
        self.class_to_idx = {}  # Mapping of class names to indices
        self._prepare_dataset()  # Prepare dataset by loading image paths and labels

    def _prepare_dataset(self):
        """
        Scan the dataset directory and prepare the image paths along with their labels.
        Binary labels are determined by specific class names.
        """
        classes = sorted(os.listdir(self.root_dir))
        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx  # Store class-to-index mapping
                for img_name in os.listdir(class_dir):
                    # Check if the file is an image
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                        img_path = os.path.join(class_dir, img_name)
                        self.image_paths.append(img_path)
                        self.subtype_labels.append(idx)
                        # Assign binary labels based on class name
                        if class_name in ['A', 'F', 'PT', 'TA']:
                            self.binary_labels.append(0)  # Classifies as 0 for these classes
                        else:
                            self.binary_labels.append(1)  # Classifies as 1 for other classes

    def __len__(self):
        """
        Return the total number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load and return an image along with its binary and subtype labels.
        """
        image = Image.open(self.image_paths[idx]).convert('RGB')
        subtype_label = self.subtype_labels[idx]
        binary_label = self.binary_labels[idx]
        if self.transform:
            image = self.transform(image)  # Apply transformations if provided
        return image, binary_label, subtype_label


def main():
    # Define paths to training and validation datasets
    train_dir = r"C:\Users\eyita\Downloads\dataset\train"
    val_dir = r"C:\Users\eyita\Downloads\dataset\validation"

    # Image transformations for the training dataset (with data augmentation)
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])

    # Image transformations for the validation dataset (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create dataset objects for training and validation
    train_dataset = TumorDataset(root_dir=train_dir, transform=train_transforms)
    val_dataset = TumorDataset(root_dir=val_dir, transform=val_transforms)

    # DataLoader for loading datasets in batches
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define a model for multi-output classification (binary and subtype classification)
    class MultiOutputModel(nn.Module):
        def __init__(self, num_subtypes):
            super(MultiOutputModel, self).__init__()
            self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  # Load pre-trained ResNet18
            num_features = self.base_model.fc.in_features  # Number of features from the final layer
            self.base_model.fc = nn.Identity()  # Replace final layer with identity (no operation)

            # Define two output layers: one for binary classification, one for subtype classification
            self.fc_binary = nn.Linear(num_features, 1)
            self.fc_subtype = nn.Linear(num_features, num_subtypes)

        def forward(self, x):
            x = self.base_model(x)  # Extract features using the ResNet18 base model
            binary_output = torch.sigmoid(self.fc_binary(x))  # Binary output with sigmoid activation
            subtype_output = self.fc_subtype(x)  # Subtype output (raw logits)
            return binary_output, subtype_output

    # Use GPU if available, otherwise fallback to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model and move it to the appropriate device
    num_subtypes = len(train_dataset.class_to_idx)
    model = MultiOutputModel(num_subtypes=num_subtypes).to(device)

    # Define loss functions: Binary Cross-Entropy for binary classification and Cross-Entropy for subtypes
    criterion_binary = nn.BCELoss()
    criterion_subtype = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Number of epochs for training
    num_epochs = 25

    # Lists to store training and validation metrics
    train_losses = []
    val_losses = []
    train_accs_binary = []
    val_accs_binary = []
    train_accs_subtype = []
    val_accs_subtype = []

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_binary = 0
        correct_subtype = 0
        total = 0

        for images, binary_labels, subtype_labels in train_loader:
            images = images.to(device)
            binary_labels = binary_labels.to(device, dtype=torch.float32)
            subtype_labels = subtype_labels.to(device)

            optimizer.zero_grad()  # Zero out gradients
            outputs_binary, outputs_subtype = model(images)  # Forward pass

            outputs_binary = outputs_binary.squeeze()  # Remove extra dimensions for binary output
            loss_binary = criterion_binary(outputs_binary, binary_labels)  # Binary classification loss
            loss_subtype = criterion_subtype(outputs_subtype, subtype_labels)  # Subtype classification loss
            loss = loss_binary + loss_subtype  # Total loss

            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters

            running_loss += loss.item() * images.size(0)
            total += binary_labels.size(0)

            # Compute binary classification accuracy
            predicted_binary = (outputs_binary >= 0.5).long()
            correct_binary += (predicted_binary == binary_labels.long()).sum().item()

            # Compute subtype classification accuracy
            _, predicted_sub = torch.max(outputs_subtype, 1)
            correct_subtype += (predicted_sub == subtype_labels).sum().item()

        # Compute average loss and accuracy for the epoch
        epoch_loss = running_loss / total
        epoch_acc_binary = correct_binary / total
        epoch_acc_subtype = correct_subtype / total

        train_losses.append(epoch_loss)
        train_accs_binary.append(epoch_acc_binary)
        train_accs_subtype.append(epoch_acc_subtype)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Loss: {epoch_loss:.4f}, '
              f'Binary Acc: {epoch_acc_binary:.4f}, '
              f'Subtype Acc: {epoch_acc_subtype:.4f}')

        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct_binary_val = 0
        correct_subtype_val = 0
        total_val = 0

  # Perform validation loop without computing gradients
        with torch.no_grad():
            for images, binary_labels, subtype_labels in val_loader:
                images = images.to(device)
                binary_labels = binary_labels.to(device, dtype=torch.float32)
                subtype_labels = subtype_labels.to(device)

                outputs_binary, outputs_subtype = model(images)
                outputs_binary = outputs_binary.squeeze()
                loss_binary = criterion_binary(outputs_binary, binary_labels)
                loss_subtype = criterion_subtype(outputs_subtype, subtype_labels)
                loss = loss_binary + loss_subtype

                val_loss += loss.item() * images.size(0)
                total_val += binary_labels.size(0)

                predicted_binary = (outputs_binary >= 0.5).long()
                correct_binary_val += (predicted_binary == binary_labels.long()).sum().item()

                _, predicted_sub = torch.max(outputs_subtype, 1)
                correct_subtype_val += (predicted_sub == subtype_labels).sum().item()

        val_loss_epoch = val_loss / total_val
        val_acc_binary_ = correct_binary_val / total_val
        val_acc_subtype_ = correct_subtype_val / total_val

        val_losses.append(val_loss_epoch)
        val_accs_binary.append(val_acc_binary_)
        val_accs_subtype.append(val_acc_subtype_)
  # Print the validation metrics for the epoch
        print(f'Validation Loss: {val_loss_epoch:.4f}, '
              f'Binary Val Acc: {val_acc_binary_:.4f}, '
              f'Subtype Val Acc: {val_acc_subtype_:.4f}')

    # Save model and mappings
    torch.save(model.state_dict(), 'multi_output_model.pth')
    print("Model saved as 'multi_output_model.pth'")

    with open('class_to_idx.pkl', 'wb') as f:
        pickle.dump(train_dataset.class_to_idx, f)
    print("Class to index mapping saved as 'class_to_idx.pkl'")

# Plot the training and validation metrics over epochs
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 8))

  # Plot binary classification accuracy for training and validation
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, train_accs_binary, label='Train Binary Acc')
    plt.plot(epochs_range, val_accs_binary, label='Val Binary Acc')
    plt.legend()
    plt.title('Binary Classification Accuracy')
# Plot subtype classification accuracy for training and validation
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, train_accs_subtype, label='Train Subtype Acc')
    plt.plot(epochs_range, val_accs_subtype, label='Val Subtype Acc')
    plt.legend()
    plt.title('Subtype Classification Accuracy')
 # Plot the training and validation loss over epochs
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
  # Adjust layout to prevent overlapping of plots
    plt.tight_layout()
# Save the plot to a file
    plt.savefig('training_metrics_pytorch.png')
    print("Training metrics plot saved as 'training_metrics_pytorch.png'")

if __name__ == '__main__':
    main()