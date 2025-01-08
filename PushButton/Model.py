import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F

# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, lstm_hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(lstm_hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, lstm_hidden_dim)
        weights = self.attention_weights(lstm_output)  # shape: (batch_size, seq_len, 1)
        weights = torch.softmax(weights, dim=1)  # Normalize weights across sequence length
        weighted_output = lstm_output * weights  # Apply weights to LSTM outputs
        context_vector = weighted_output.sum(dim=1)  # Summarize over the sequence length
        return context_vector, weights

# Define CNN + LSTM Model with Attention Mechanism
class ResNetLSTMWithAttention(nn.Module):
    def __init__(self, lstm_hidden_dim=128, lstm_layers=1):
        super(ResNetLSTMWithAttention, self).__init__()

        # Pretrained ResNet model as feature extractor
        resnet = models.resnet34(weights = models.ResNet34_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # LSTM to process frame sequence
        self.lstm = nn.LSTM(input_size=resnet.fc.in_features, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)

        # Attention mechanism
        self.attention = Attention(lstm_hidden_dim)

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

        # Attention mechanism
        context_vector, attention_weights = self.attention(lstm_out)

        # Classification
        out = self.fc(context_vector)
        out = self.sigmoid(out)
        return out, attention_weights

# Define CNN + LSTM Model with ResNet Backbone
class ResNetLSTM(nn.Module):
    def __init__(self, lstm_hidden_dim=128, lstm_layers=1):
        super(ResNetLSTM, self).__init__()

        # Pretrained ResNet model as feature extractor
        resnet = models.resnet18(weights = models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        # LSTM to process frame sequence
        self.lstm = nn.LSTM(input_size=resnet.fc.in_features, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)

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
    

# CNN for Feature Extraction
class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim=256):
        super(CNNFeatureExtractor, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(32 * 37 * 62, output_dim),  # Adjust based on ROI size
            nn.ReLU()
        )

    def forward(self, x):
        return self.cnn(x)

# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, lstm_hidden_dim):
        super(Attention, self).__init__()
        self.attention_weights = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, hidden_dim)
        weights = self.attention_weights(lstm_output)  # (batch_size, seq_len, 1)
        weights = F.softmax(weights, dim=1)  # Normalize across the sequence
        context = torch.sum(weights * lstm_output, dim=1)  # Weighted sum
        return context, weights

# CNN + LSTM + Attention Model
class CNNLSTMAttention(nn.Module):
    def __init__(self, cnn_output_dim=256, lstm_hidden_dim=128, lstm_layers=1):
        super(CNNLSTMAttention, self).__init__()
        self.cnn = CNNFeatureExtractor(output_dim=cnn_output_dim)
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.attention = Attention(lstm_hidden_dim)
        self.fc = nn.Linear(lstm_hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)  # Flatten the sequence dimension
        cnn_features = self.cnn(x)  # Extract CNN features for each frame
        cnn_features = cnn_features.view(batch_size, seq_len, -1)  # Reshape for LSTM
        lstm_output, _ = self.lstm(cnn_features)  # Process with LSTM
        context, attention_weights = self.attention(lstm_output)  # Apply Attention
        out = self.fc(context)  # Classification
        out = self.sigmoid(out)  # Binary probability
        return out, attention_weights