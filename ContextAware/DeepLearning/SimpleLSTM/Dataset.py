from torch.utils.data import Dataset, DataLoader
import torch 
import os
import numpy as np
from DDBSCAN import Raster_DBSCAN
from Utils import get_trajs_from_Kalman_out
import random
import h5py

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
        
        # If no trajectories found, return dummy tensors
        if not trajs:
            return (
                torch.zeros(self.sequence_length, dtype=torch.float32),  # Sequence tensor
                torch.zeros(self.sequence_length, dtype=torch.bool),     # Mask tensor
                torch.zeros(1, dtype=torch.float32)    # Target tensor
            )
        
        # Process the first valid trajectory
        traj = trajs[0]  # Take the first trajectory
        
        # Check if trajectory has enough points
        if len(traj) < self.sequence_length + 1:  # Need at least 11 points for 10 input + 1 target
            return (
                torch.zeros(self.sequence_length, dtype=torch.float32),
                torch.zeros(self.sequence_length, dtype=torch.bool),
                torch.zeros(1, dtype=torch.float32)
            )
        
        # Take last 11 points, first 10 as input, last one as target
        input_points = traj[-(self.sequence_length+1):-1, 1]  # Position values from the trajectory
        target_point = traj[-1, 1]      # Last position value
        
        # Create random occlusion mask
        mask = np.ones(self.sequence_length, dtype=bool)
        occluded_input = input_points.copy()
        
        # Apply random occlusions
        for i in range(self.sequence_length):
            if np.random.random() < 0.2:  # 20% occlusion rate
                mask[i] = False
                occluded_input[i] = -1  # Mark as occluded
        
        # Convert to tensors
        sequence_tensor = torch.tensor(occluded_input, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        target_tensor = torch.tensor([target_point], dtype=torch.float32)
        
        return sequence_tensor, mask_tensor, target_tensor
    

class PicklableH5Dataset(Dataset):
    """
    A wrapper dataset that loads data from the H5 file but is picklable for multiprocessing
    with additional trajectory variation augmentation
    """
    def __init__(self, h5_path, occlusion_rate=0.2, 
                 position_noise_std=1.0, time_dilation_range=(0.9, 1.1)):
        self.h5_path = h5_path
        self.occlusion_rate = occlusion_rate
        self.position_noise_std = position_noise_std
        self.time_dilation_range = time_dilation_range
        
        # Read all data into memory
        with h5py.File(h5_path, 'r') as f:
            self.sequences = f['input_sequences'][:]
            self.masks = f['masks'][:]
            self.targets = f['targets'][:]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx].copy()
        mask = self.masks[idx].copy()
        target = self.targets[idx]
        
        # Apply trajectory variation augmentations
        # 1. Add position noise
        if self.position_noise_std > 0:
            noise = np.random.normal(0, self.position_noise_std, size=sequence.shape)
            sequence = sequence + noise
        
        # 2. Apply time dilation/compression (simulate varying speeds)
        if random.random() < 0.5 and self.time_dilation_range[0] < self.time_dilation_range[1]:
            # Generate time dilation factor
            dilation = random.uniform(self.time_dilation_range[0], self.time_dilation_range[1])
            
            # Calculate trend from sequence to target
            seq_len = len(sequence)
            start_pos = sequence[0]
            end_pos = sequence[-1]
            target_pos = target
            
            # Apply dilation to create a new sequence that still ends at the target
            trend = (end_pos - start_pos) / seq_len
            dilated_trend = trend * dilation
            
            # Create new sequence with the dilated trend
            new_sequence = np.zeros_like(sequence)
            for i in range(seq_len):
                new_sequence[i] = start_pos + dilated_trend * i
            
            # Adjust target based on the dilation
            new_target = end_pos + dilated_trend
            
            # Add some non-linear variation
            if random.random() < 0.3:
                # Add a small sinusoidal component
                amplitude = abs(end_pos - start_pos) * 0.05  # 5% of total movement
                frequency = random.uniform(0.5, 2.0)
                phase = random.uniform(0, 2 * np.pi)
                for i in range(seq_len):
                    t = i / seq_len
                    new_sequence[i] += amplitude * np.sin(2 * np.pi * frequency * t + phase)
            
            sequence = new_sequence
            target = new_target
        
        # 3. Apply random occlusion
        occluded_sequence = sequence.copy()
        for i in range(len(sequence)):
            if random.random() < self.occlusion_rate:
                occluded_sequence[i] = -1
                mask[i] = False
        
        # Convert to tensors
        sequence_tensor = torch.tensor(occluded_sequence, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        target_tensor = torch.tensor([target], dtype=torch.float32)
        
        return sequence_tensor, mask_tensor, target_tensor