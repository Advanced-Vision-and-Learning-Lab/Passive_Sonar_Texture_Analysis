
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


df = pd.read_pickle('../datasets_modulated/features_segmented_dataframe_V4.pkl')
# Separate features and labels
X_stat = np.vstack(df['Statistical Features'].values).astype(np.float32)
y = df['Label'].values

# Encode string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

def custom_train_val_test_split(X, y, train_size, val_size, test_size):
    unique_classes = np.unique(y)
    X_train, X_val, X_test = [], [], []
    y_train, y_val, y_test = [], [], []
    
    for cls in unique_classes:
        # Find indexes of the current class
        class_idx = np.where(y == cls)[0]
        # Calculate split sizes for the current class
        train_end = int(len(class_idx) * train_size)
        val_end = train_end + int(len(class_idx) * val_size)
        
        # Split the data and labels for the current class
        X_train.extend(X[class_idx[:train_end]])
        y_train.extend(y[class_idx[:train_end]])
        
        X_val.extend(X[class_idx[train_end:val_end]])
        y_val.extend(y[class_idx[train_end:val_end]])
        
        X_test.extend(X[class_idx[val_end:]])
        y_test.extend(y[class_idx[val_end:]])
    
    return np.array(X_train), np.array(X_val), np.array(X_test), np.array(y_train), np.array(y_val), np.array(y_test)


train_size, val_size, test_size = 0.7, 0.15, 0.15
X_train_stat, X_val_stat, X_test_stat, y_train, y_val, y_test = custom_train_val_test_split(X_stat, y_encoded, train_size, val_size, test_size)

# Standardize the features
scaler = StandardScaler()
X_train_stat = scaler.fit_transform(X_train_stat)
X_val_stat = scaler.transform(X_val_stat)
X_test_stat = scaler.transform(X_test_stat)

# Print dataset info
print("Training set size:", X_train_stat.shape[0])
print("Validation set size:", X_val_stat.shape[0])
print("Test set size:", X_test_stat.shape[0])

# Decode labels to original ship types for readability
ship_types = label_encoder.inverse_transform(np.unique(y_encoded))

# Print class distribution in training set
train_label_counts = pd.Series(label_encoder.inverse_transform(y_train)).value_counts()
print("Class distribution in training set:\n", train_label_counts)

# Print class distribution in val set
val_label_counts = pd.Series(label_encoder.inverse_transform(y_val)).value_counts()
print("Class distribution in val set:\n", val_label_counts)

# Print class distribution in test set
test_label_counts = pd.Series(label_encoder.inverse_transform(y_test)).value_counts()
print("Class distribution in test set:\n", test_label_counts)

# Convert to PyTorch tensors
train_data = TensorDataset(torch.from_numpy(X_train_stat), torch.from_numpy(y_train))
val_data = TensorDataset(torch.from_numpy(X_val_stat), torch.from_numpy(y_val))
test_data = TensorDataset(torch.from_numpy(X_test_stat), torch.from_numpy(y_test))

batch_size = 32  
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)
test_loader = DataLoader(test_data, batch_size=batch_size)

num_features = X_train_stat.shape[1]
num_classes = len(df['Label'].unique())
print("Number of input features:", num_features, "output classes:", num_classes)


class SimpleFCN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFCN, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        return self.fc(x)

def train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')  
    patience, patience_limit = 0, 50 

    for epoch in range(num_epochs):
        total_loss = 0
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
        avg_val_loss = total_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Monitor validation loss for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience = 0  # Reset patience
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"Early stopping triggered at epoch {epoch + 1}. " +
                      f"Validation loss did not improve for {patience_limit} consecutive epochs.")
                break  

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    return train_losses, val_losses


def get_predictions_and_accuracy(model, data_loader, device):
    model.eval()
    all_preds = []
    true_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    return all_preds, true_labels, accuracy


def run_experiment(run_id, device='cuda'):
    
    model = SimpleFCN(num_features, num_classes)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    num_epochs = 2500
    train_losses, val_losses = train_and_validate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    
    # Plot training and validation loss
    plt.figure(figsize=(4, 3))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Loss Plot Stats - Run {run_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(f'../figures_modulated/loss_plot_stats_run_{run_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

    test_preds, test_labels, test_accuracy = get_predictions_and_accuracy(model, test_loader, device)

    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix Stats - Test Data Run {run_id}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'../figures_modulated/confusion_matrix_stats_run_{run_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Run {run_id}: Test Accuracy: {test_accuracy:.4f}')
    return test_accuracy, cm

accuracies = []
for run_id in range(1, 4):
    accuracy, _ = run_experiment(run_id, device='cuda')
    accuracies.append(accuracy)

average_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print(f'Average Test Accuracy: {average_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%')
print("\nModel A (STATS) Training Done.")
