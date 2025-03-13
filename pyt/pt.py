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

class TumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.binary_labels = []
        self.subtype_labels = []
        self.class_to_idx = {}
        self._prepare_dataset()

    def _prepare_dataset(self):
        classes = sorted(os.listdir(self.root_dir))
        for idx, class_name in enumerate(classes):
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = idx
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                        img_path = os.path.join(class_dir, img_name)
                        self.image_paths.append(img_path)
                        self.subtype_labels.append(idx)
                        # Adjust class names as per your dataset
                        if class_name in ['A', 'F', 'PT', 'TA']:
                            self.binary_labels.append(0)
                        else:
                            self.binary_labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        subtype_label = self.subtype_labels[idx]
        binary_label = self.binary_labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, binary_label, subtype_label


def main():
    train_dir = r"C:\Users\eyita\Downloads\dataset\train"
    val_dir = r"C:\Users\eyita\Downloads\dataset\validation"

    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = TumorDataset(root_dir=train_dir, transform=train_transforms)
    val_dataset = TumorDataset(root_dir=val_dir, transform=val_transforms)

    batch_size = 32
    # If problems persist, try num_workers=0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    class MultiOutputModel(nn.Module):
        def __init__(self, num_subtypes):
            super(MultiOutputModel, self).__init__()
            self.base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            num_features = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()

            self.fc_binary = nn.Linear(num_features, 1)
            self.fc_subtype = nn.Linear(num_features, num_subtypes)

        def forward(self, x):
            x = self.base_model(x)
            binary_output = torch.sigmoid(self.fc_binary(x))
            subtype_output = self.fc_subtype(x)
            return binary_output, subtype_output

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_subtypes = len(train_dataset.class_to_idx)
    model = MultiOutputModel(num_subtypes=num_subtypes).to(device)

    criterion_binary = nn.BCELoss()
    criterion_subtype = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 25

    train_losses = []
    val_losses = []
    train_accs_binary = []
    val_accs_binary = []
    train_accs_subtype = []
    val_accs_subtype = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_binary = 0
        correct_subtype = 0
        total = 0

        for images, binary_labels, subtype_labels in train_loader:
            images = images.to(device)
            binary_labels = binary_labels.to(device, dtype=torch.float32)
            subtype_labels = subtype_labels.to(device)

            optimizer.zero_grad()
            outputs_binary, outputs_subtype = model(images)

            outputs_binary = outputs_binary.squeeze()
            loss_binary = criterion_binary(outputs_binary, binary_labels)
            loss_subtype = criterion_subtype(outputs_subtype, subtype_labels)
            loss = loss_binary + loss_subtype

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            total += binary_labels.size(0)

            predicted_binary = (outputs_binary >= 0.5).long()
            correct_binary += (predicted_binary == binary_labels.long()).sum().item()

            _, predicted_sub = torch.max(outputs_subtype, 1)
            correct_subtype += (predicted_sub == subtype_labels).sum().item()

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

        model.eval()
        val_loss = 0.0
        correct_binary_val = 0
        correct_subtype_val = 0
        total_val = 0

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

        print(f'Validation Loss: {val_loss_epoch:.4f}, '
              f'Binary Val Acc: {val_acc_binary_:.4f}, '
              f'Subtype Val Acc: {val_acc_subtype_:.4f}')

    # Save model and mappings
    torch.save(model.state_dict(), 'multi_output_model.pth')
    print("Model saved as 'multi_output_model.pth'")

    with open('class_to_idx.pkl', 'wb') as f:
        pickle.dump(train_dataset.class_to_idx, f)
    print("Class to index mapping saved as 'class_to_idx.pkl'")

    # Plot metrics
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(12, 8))

    # Binary classification accuracy
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, train_accs_binary, label='Train Binary Acc')
    plt.plot(epochs_range, val_accs_binary, label='Val Binary Acc')
    plt.legend()
    plt.title('Binary Classification Accuracy')

    # Subtype classification accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, train_accs_subtype, label='Train Subtype Acc')
    plt.plot(epochs_range, val_accs_subtype, label='Val Subtype Acc')
    plt.legend()
    plt.title('Subtype Classification Accuracy')

    # Training and validation loss
    plt.subplot(2, 1, 2)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Val Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.tight_layout()
    plt.savefig('training_metrics_pytorch.png')
    plt.close()
    print("Training metrics plot saved as 'training_metrics_pytorch.png'")

if __name__ == '__main__':
    main()
