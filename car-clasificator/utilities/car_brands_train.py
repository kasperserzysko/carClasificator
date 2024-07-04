import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, precision_score
from tqdm import tqdm

# Parameters
data_dir = '../data'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)
batch_size = 32
img_height = 224
img_width = 224
epochs = 40
learning_rate = 0.001
validation_split = 0.2
model_path = os.path.join(output_dir, 'car_brand_classifier_resnet.pth')

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((img_height, img_width)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((img_height, img_width), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(root=data_dir, transform=transform)
val_size = int(len(dataset) * validation_split)
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = models.resnet50(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': [],
               'train_f1': [], 'val_f1': [], 'train_precision': [], 'val_precision': [],
               'train_recall': [], 'val_recall': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total
        epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
        epoch_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        epoch_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        history['train_loss'].append(epoch_loss)
        history['train_accuracy'].append(epoch_accuracy)
        history['train_f1'].append(epoch_f1)
        history['train_precision'].append(epoch_precision)
        history['train_recall'].append(epoch_recall)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = correct / total
        val_f1 = f1_score(all_labels, all_preds, average='weighted')
        val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)

        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['val_f1'].append(val_f1)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Train F1: {epoch_f1:.4f}, '
              f'Val F1: {val_f1:.4f}, Train Precision: {epoch_precision:.4f}, Val Precision: {val_precision:.4f}, '
              f'Train Recall: {epoch_recall:.4f}, Val Recall: {val_recall:.4f}')

    return history


history = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
torch.save(model.state_dict(), model_path)

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(os.path.join(output_dir, 'loss.png'))

plt.subplot(2, 2, 2)
plt.plot(history['train_accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig(os.path.join(output_dir, 'accuracy.png'))

plt.subplot(2, 2, 3)
plt.plot(history['train_f1'], label='Training F1 Score')
plt.plot(history['val_f1'], label='Validation F1 Score')
plt.title('F1 Score over epochs')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.savefig(os.path.join(output_dir, 'f1_score.png'))

plt.subplot(2, 2, 4)
plt.plot(history['train_precision'], label='Training Precision')
plt.plot(history['val_precision'], label='Validation Precision')
plt.title('Precision over epochs')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.savefig(os.path.join(output_dir, 'precision.png'))

plt.tight_layout()
plt.show()
