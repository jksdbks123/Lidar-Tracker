import pandas as pd
import numpy as np
import torch
from torch._C import device
import torch.nn as nn 
import torch.nn.functional as F
import os 

class TrackingNet(nn.Module):
    
    def __init__(self):
        super (TrackingNet,self).__init__()
        
        self.fc1 = nn.Linear(4,128)
        self.fc2 = nn.Linear(128,256)
        self.fc3 = nn.Linear(256,256)
        self.fc4 = nn.Linear(256,256)
        self.fc5 = nn.Linear(256,2)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self,x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc4(x))
        x = self.dropout(x)
        x = self.fc5(x)
        
        return x
        

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# test_tensor = torch.tensor([[-11.7860887466129,-24.1342180758378,-1,-1]])
# model = TrackingNet()
# model.load_state_dict(torch.load('./PosteriorModel/saved_model.pth', map_location=device))
# model.eval()
# print(model(test_tensor))


