import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
# Define CNN + LSTM Model with ResNet Backbone
class ResNetLSTM(nn.Module):
    def __init__(self, lstm_hidden_dim=128, lstm_layers=1):
        super(ResNetLSTM, self).__init__()

        # Pretrained ResNet model as feature extractor
        resnet = models.resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # LSTM to process frame sequence
        self.lstm = nn.LSTM(input_size=512, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)

        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)

        # ResNet feature extraction
        features = self.feature_extractor(x)
        features = features.view(batch_size, seq_len, -1)

        # LSTM sequence processing
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]  # Take the last output from the sequence

        # Classification
        out = self.fc(lstm_out)
        out = self.sigmoid(out)
        return out