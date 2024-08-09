import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json

input_size = 1024
hidden_size = 500
num_classes = 1
num_epoch = 50
learning_rate = 0.01
BATCH_SIZE = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Neural Network
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
