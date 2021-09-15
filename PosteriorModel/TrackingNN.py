import pandas as pd
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F
import os 
class TrackingNet(nn.Module):
    
    def __init__(self):
        super (TrackingNet,self).__init__()
        
        self.fc1 = nn.Linear(4,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,4)
        
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

test_tensor = torch.rand((2,4))
model = TrackingNet()
print(os.getcwd())
model.load_state_dict(torch.load('saved_model.pth'))
model.eval()
