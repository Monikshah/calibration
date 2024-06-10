import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

input_size = 1024
hidden_size = 500
num_classes = 2
num_epoch = 2
batch_size = 50
learning_rate = 0.01
BATCH_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Neural Network
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):

        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

        # self.train_loader = DataLoader(
        # dataset=train_dataset, batch_size=batch_size, shuffle=True
        # )
        # self.test_loader = DataLoader(
        # dataset=test_dataset, batch_size=batch_size, shuffle=False
        # )

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
