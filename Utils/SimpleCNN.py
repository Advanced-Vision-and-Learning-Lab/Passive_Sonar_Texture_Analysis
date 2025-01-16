import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        # Initialize fc1 later using a placeholder for the flattened size
        self.fc1 = nn.Linear(16384, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def _initialize_fc1(self, x):
        # Determine the flattened size dynamically
        with torch.no_grad():
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            flattened_size = x.view(x.size(0), -1).size(1)
        self.fc1 = nn.Linear(flattened_size, 128)

    def forward(self, x):
        if self.fc1 is None:
            self._initialize_fc1(x)
        
        xx = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(xx)))
        flattened_x = x.view(x.size(0), -1)  # Flatten the tensor
        x = F.relu(self.fc1(flattened_x))
        x = self.fc2(x)
        return xx, x
