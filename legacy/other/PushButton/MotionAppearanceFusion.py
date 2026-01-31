import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Linear(32 * 50 * 75, output_dim),  # Adjust based on input dimensions
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

# CNN-LSTM Model
class CNNLSTMAttention(nn.Module):
    def __init__(self, cnn_output_dim=512, lstm_hidden_dim=256, lstm_layers=1):
        super(CNNLSTMAttention, self).__init__()
        self.cnn = CNNFeatureExtractor(output_dim=cnn_output_dim)
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        self.attention = Attention(lstm_hidden_dim)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)  # Flatten the sequence dimension
        cnn_features = self.cnn(x)  # Extract CNN features for each frame
        cnn_features = cnn_features.view(batch_size, seq_len, -1)  # Reshape for LSTM
        lstm_output, _ = self.lstm(cnn_features)  # Process with LSTM
        context, attention_weights = self.attention(lstm_output)  # Apply Attention
        return context

# Two-Stream Model
class TwoStreamModel(nn.Module):
    def __init__(self, cnn_output_dim=512, lstm_hidden_dim=256, lstm_layers=1):
        super(TwoStreamModel, self).__init__()
        self.rgb_stream = CNNLSTMAttention(cnn_output_dim, lstm_hidden_dim, lstm_layers)
        self.flow_stream = CNNLSTMAttention(cnn_output_dim, lstm_hidden_dim, lstm_layers)
        self.fc = nn.Linear(lstm_hidden_dim * 2, 1)  # Combine features from both streams
        self.sigmoid = nn.Sigmoid()

    def forward(self, rgb_input, flow_input):
        rgb_features = self.rgb_stream(rgb_input)
        flow_features = self.flow_stream(flow_input)
        combined_features = torch.cat([rgb_features, flow_features], dim=1)  # Concatenate
        out = self.fc(combined_features)  # Classification
        out = self.sigmoid(out)
        return out
