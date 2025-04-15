from torch.utils.data import Dataset, DataLoader
import torch 
import os
import numpy as np
from DDBSCAN import Raster_DBSCAN
from Utils import get_trajs_from_Kalman_out
import random

class TrajDataset(Dataset):
    def __init__(self, data_dir, time_span, sequence_length=10, occlusion_rate=0.2):
        self.data_dir = data_dir
        self.time_span = time_span
        self.occlusion_rate = occlusion_rate
        self.sequence_length = sequence_length
        # Determine which folder to use based on whether it's training or validation
        self.folder_path = data_dir
        self.db = Raster_DBSCAN(Td_map_szie=(200,100), window_size=[5,5], eps=1, min_samples=1)

        # Get list of file names (assuming they're numbered consistently across subfolders)
        self.file_names = [f for f in os.listdir(os.path.join(self.folder_path, 'target')) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        
        # Load target (shape: 200, future_length)
        target_path = os.path.join(self.folder_path, 'target', file_name)
        target = np.load(target_path)
        target = target[:, :self.time_span]
        Labels = self.db.fit_predict(target, target > 0)
        
        # trajs: list - each element in the list is a np.array of shape (n_frames, 2) : [(t, pos)]
        trajs = get_trajs_from_Kalman_out(Labels, max_prediction_count=2)
        
        # Process trajectories to create training samples with occlusion augmentation
        processed_samples = []
        
        for traj in trajs:
            if len(traj) < self.sequence_length + 1:
                continue  # Skip if trajectory is too short
            
            # Find valid sequence windows within the trajectory
            for start_idx in range(len(traj) - self.sequence_length):
                # Extract sequence and target
                seq = traj[start_idx:start_idx+self.sequence_length]
                next_pos = traj[start_idx+self.sequence_length][1]  # Position part of (t, pos)
                
                # Apply random occlusion
                mask = np.ones(self.sequence_length, dtype=bool)
                occluded_seq = seq.copy()
                
                # Randomly occlude some positions based on occlusion_rate
                for i in range(self.sequence_length):
                    if random.random() < self.occlusion_rate:
                        # Mark as occluded - set position to None (represented as -1)
                        occluded_seq[i, 1] = -1
                        mask[i] = False
                
                # Only use sequences with at least half non-occluded frames
                if np.sum(mask) >= self.sequence_length // 2:
                    processed_samples.append({
                        'sequence': occluded_seq[:, 1],  # Position values
                        'mask': mask,
                        'target': next_pos
                    })
        
        # If no valid samples were found, return a dummy sample
        if not processed_samples:
            return {
                'sequence': np.zeros(self.sequence_length),
                'mask': np.zeros(self.sequence_length, dtype=bool),
                'target': 0.0
            }
        
        # Return a randomly selected valid sample
        sample = random.choice(processed_samples)
        
        # Convert to tensors
        sequence_tensor = torch.tensor(sample['sequence'], dtype=torch.float32)
        mask_tensor = torch.tensor(sample['mask'], dtype=torch.bool)
        target_tensor = torch.tensor([sample['target']], dtype=torch.float32)
        
        return sequence_tensor, mask_tensor, target_tensor