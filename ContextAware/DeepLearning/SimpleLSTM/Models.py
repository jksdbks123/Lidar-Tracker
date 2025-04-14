
import torch.nn as nn

class SimpleLSTMPredictor(nn.Module):
    def __init__(self, input_size=5, hidden_size=32, output_size=1):
        """
        Simple LSTM model for predicting the next position based on sequence of past positions
        
        Parameters:
        -----------
        input_size: int
            Number of past positions to use as input
        hidden_size: int
            Size of LSTM hidden layer
        output_size: int
            Size of output (1 for position prediction)
        """
        super(SimpleLSTMPredictor, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=1,            # 1D input (position)
            hidden_size=hidden_size, 
            num_layers=1,
            batch_first=True
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, 1)
        lstm_out, _ = self.lstm(x)
        
        # Use only the last output from LSTM
        lstm_out = lstm_out[:, -1, :]
        
        # Pass through fully connected layer to get prediction
        prediction = self.fc(lstm_out)
        
        return prediction