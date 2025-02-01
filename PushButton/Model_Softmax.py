import torch.nn as nn
from torchvision import models
import torch
import torch.nn.functional as F

# Attention-based CNN + LSTM Model
class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(input_dim + hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, lstm_output, hidden_state):
        # Concatenate hidden state with LSTM outputs across the sequence
        hidden_state = hidden_state.squeeze(0).unsqueeze(1).repeat(1, lstm_output.size(1), 1)
        combined = torch.cat((lstm_output, hidden_state), dim=2)
        attention_scores = self.attention(combined)
        attention_weights = self.softmax(attention_scores)
        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights

class CNNLSTMWithAttention(nn.Module):
    def __init__(self, cnn_output_dim=256, lstm_hidden_dim=128, lstm_layers=1, num_classes=3):
        super(CNNLSTMWithAttention, self).__init__()

        # CNN to extract features from each frame
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout(0.3),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout(0.5),
        #     nn.Flatten(),
        #     nn.Linear(32 * int(250/4) * int(150/4), cnn_output_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(64 * int(250/8) * int(150/8), cnn_output_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
            
        )

        # LSTM to process frame sequence
        self.lstm = nn.LSTM(input_size=cnn_output_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)

        # Attention mechanism
        self.attention = Attention(input_dim=cnn_output_dim, hidden_dim=lstm_hidden_dim)

        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)

        # CNN feature extraction
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)

        # LSTM sequence processing
        lstm_out, (hidden_state, _) = self.lstm(cnn_features)

        # Attention mechanism
        context, attention_weights = self.attention(lstm_out, hidden_state)

        # Classification
        out = self.fc(context)
        out = self.softmax(out)
        return out, attention_weights
    
class CNNLSTM(nn.Module):
    def __init__(self, cnn_output_dim=256, lstm_hidden_dim=128, lstm_layers=1, num_classes=3):
        super(CNNLSTM, self).__init__()

        # CNN to extract features from each frame
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout(0.3),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout(0.5),
        #     nn.Flatten(),
        #     nn.Linear(32 * int(250/4) * int(150/4), cnn_output_dim),
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(64 * int(250/8) * int(150/8), cnn_output_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
            
        )

        # LSTM to process frame sequence
        self.lstm = nn.LSTM(input_size=cnn_output_dim, hidden_size=lstm_hidden_dim, num_layers=lstm_layers, batch_first=True)
        

        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)

        # CNN feature extraction
        cnn_features = self.cnn(x)
        cnn_features = cnn_features.view(batch_size, seq_len, -1)

        # LSTM sequence processing
        lstm_out, (hidden_state, _) = self.lstm(cnn_features)
        # Classification
        out = self.fc(lstm_out[:,-1,:])
        out = self.softmax(out)

        return out
    
# Focal Loss for Multi-Class Classification
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.15, gamma=2, num_classes=3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs, targets):
        inputs = torch.log_softmax(inputs, dim=1)
        targets_one_hot = torch.eye(self.num_classes).to(inputs.device)[targets.to(torch.int)]
        pt = torch.exp(inputs) * targets_one_hot  # Probabilities for correct classes
        focal_loss = -self.alpha * ((1 - pt) ** self.gamma) * inputs * targets_one_hot
        return focal_loss.sum(dim=1).mean()