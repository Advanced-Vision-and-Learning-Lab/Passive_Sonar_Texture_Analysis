import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.nn.functional as F
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

from torch.distributions import Normal

class SimpleFCNWithKLDivergence(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SimpleFCNWithKLDivergence, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)


    def forward(self, x):
        x = self.fc(x)
        # Calculate KL Divergence here for regularization
        # feature vector to regularize to be close to standard normal distribution
        feature_distribution = Normal(x.mean(dim=0), x.std(dim=0))
        standard_normal_distribution = Normal(torch.zeros_like(x.mean(dim=0)), torch.ones_like(x.std(dim=0)))
        kl_divergence = torch.distributions.kl_divergence(feature_distribution, standard_normal_distribution).mean()

        # Return the logits and KL divergence for loss calculation
        return x, kl_divergence

def custom_loss_function(output, target, kl_divergence, lambda_kl=0.01):
    classification_loss = F.cross_entropy(output, target)
    return classification_loss, kl_divergence

def train_and_validate_model(model, train_loader, val_loader, optimizer, num_epochs, device, lambda_kl):
    model.to(device)
    train_losses, val_losses = [], []
    train_cls_losses, train_reg_losses = [], []
    val_cls_losses, val_reg_losses = [], []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_cls_loss = 0
        total_reg_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs, kl_divergence = model(inputs)
            cls_loss, reg_loss = custom_loss_function(outputs, labels, kl_divergence, lambda_kl)
            total_loss = cls_loss + lambda_kl * reg_loss
            
            total_loss.backward()
            optimizer.step()
            total_cls_loss += cls_loss.item()
            total_reg_loss += reg_loss.item()
            
        avg_train_cls_loss = total_cls_loss / len(train_loader)
        avg_train_reg_loss = total_reg_loss / len(train_loader)
        train_cls_losses.append(avg_train_cls_loss)
        train_reg_losses.append(avg_train_reg_loss)
        train_losses.append(avg_train_cls_loss + lambda_kl * avg_train_reg_loss)

        model.eval()
        total_cls_loss = 0
        total_reg_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, kl_divergence = model(inputs)
                cls_loss, reg_loss = custom_loss_function(outputs, labels, kl_divergence, lambda_kl)
                total_loss = cls_loss + lambda_kl * reg_loss
                
                total_cls_loss += cls_loss.item()
                total_reg_loss += reg_loss.item()
                
        avg_val_cls_loss = total_cls_loss / len(val_loader)
        avg_val_reg_loss = total_reg_loss / len(val_loader)
        val_cls_losses.append(avg_val_cls_loss)
        val_reg_losses.append(avg_val_reg_loss)
        val_losses.append(avg_val_cls_loss + lambda_kl * avg_val_reg_loss)

        if avg_val_cls_loss + lambda_kl * avg_val_reg_loss < best_val_loss:
            best_val_loss = avg_val_cls_loss + lambda_kl * avg_val_reg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == 50:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        if (epoch + 1) % 50 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_cls_loss + lambda_kl * avg_train_reg_loss:.4f}, Val Loss: {avg_val_cls_loss + lambda_kl * avg_val_reg_loss:.4f}')

    return train_losses, val_losses, train_cls_losses, train_reg_losses, val_cls_losses, val_reg_losses


def get_predictions_and_accuracy(model, data_loader, device):
    model.eval()
    all_preds = []
    true_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)  
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    accuracy = correct / total
    return all_preds, true_labels, accuracy


def run_experiment(run_id, device='cuda'):
    model = SimpleFCNWithKLDivergence(num_features, num_classes)
    lambda_kl = 3 
    
    optimizer = optim.Adam(model.parameters(), lr=3e-3)
    num_epochs = 2500

    train_losses, val_losses, train_cls_losses, train_reg_losses, val_cls_losses, val_reg_losses = train_and_validate_model(
        model, train_loader, val_loader, optimizer, num_epochs, device, lambda_kl
    )

    plt.figure(figsize=(5, 4))
    # Plot total training and validation loss
    plt.plot(train_losses, label='Total Training Loss', color='blue', linestyle='dashed')
    plt.plot(val_losses, label='Total Validation Loss', color='blue')
    
    # Plot classification loss for training and validation
    plt.plot(train_cls_losses, label='Training Classification Loss', color='green', linestyle='dashed')
    plt.plot(val_cls_losses, label='Validation Classification Loss', color='green')
    
    plt.plot([lambda_kl * loss for loss in train_reg_losses], label='Training KL Divergence Loss', color='red', linestyle='dashed')
    plt.plot([lambda_kl * loss for loss in val_reg_losses], label='Validation KL Divergence Loss', color='red')
        
    plt.title(f'Loss Plot Stats - Run {run_id}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(f'../figures_modulated/loss_KL_stats_run_{run_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

    test_preds, test_labels, test_accuracy = get_predictions_and_accuracy(model, test_loader, device)

    # Confusion Matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix Stats - Test Data Run {run_id}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(f'../figures_modulated/confusion_matrix_KL_stats_run_{run_id}.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f'Run {run_id}: Test Accuracy: {test_accuracy:.4f}')
    return test_accuracy, cm

accuracies = []
for run_id in range(1, 4):
    accuracy, _ = run_experiment(run_id, device='cuda')
    accuracies.append(accuracy)

average_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)
print(f'KL Average Test Accuracy: {average_accuracy*100:.2f}% ± {std_accuracy*100:.2f}%')
print("\nKL Model A (STATS) Training Done.")
