import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import os
# Custom Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from tqdm import tqdm
from TrainTrajNet import *
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    validation_loss = {}
    criterion = nn.BCELoss(reduction = 'mean').to(device)
    
    for historical_length in [5,10,20,30,40,50,70, 80 , 90, 100]:
        for future_length in [1,2,5,10,20,30,35,40,45,50]:
            
            input_size = 200  # num_lane_unit
            hidden_size = 128
            num_layers = 2
            output_size = 200 * future_length  # num_lane_unit * future_time_length (200 * 5)
            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            mocel_save_path = r"D:\TimeSpaceDiagramDataset\models"
            model_name = f'lstm_{historical_length}_{future_length}.pth'
            val_dataset = OcclusionAwareTrafficDataset(data_dir=r"D:\TimeSpaceDiagramDataset\val", historical_length=historical_length , future_length=future_length)
            val_loader = DataLoader(val_dataset, batch_size=8, num_workers = 8)
            best_model = LSTMLanePredictorWithAttention(input_size, hidden_size, num_layers, output_size)
            best_model.load_state_dict(torch.load(os.path.join(mocel_save_path, f'lstm_{historical_length}_{future_length}.pth'),weights_only=False))
            best_model.eval()
            best_model.to(device)
            val_loss = 0
            val_bar = tqdm(val_loader, desc=f'[Val]')
            for batch in val_bar:
                targets = batch['target']  
                post_occlusion = batch['post_occ_X']
                targets = targets.to(device)
                post_occlusion = post_occlusion.to(device)
                predictions, attn_weights = best_model(post_occlusion)
                val_loss += criterion(predictions, targets)
                val_bar.set_postfix({'loss': f'{val_loss.item():.4f}'})
            
            validation_loss[f'{historical_length},{future_length}'] = float(val_loss) / len(val_loader)

    # save the validation loss to a json file
    with open(r'D:\TimeSpaceDiagramDataset\models\validation_loss.json', 'w') as f:
        json.dump(validation_loss, f)
