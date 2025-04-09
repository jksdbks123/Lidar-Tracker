import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SocialLSTMDataset(Dataset):
    def __init__(self, data_dir, input_frames=10, output_frames=12, time_span=100, stride=1):
        """
        Dataset for Social LSTM training
        
        Args:
            data_dir: Directory containing the data
            input_frames: Number of input frames
            output_frames: Number of output frames
            time_span: Total time span of traffic context
            stride: Stride for generating trajectory samples
        """
        self.data_dir = data_dir
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.time_span = time_span
        self.stride = stride
        
        # Get list of file names
        self.file_names = [f for f in os.listdir(os.path.join(data_dir, 'traj_id_label')) if f.endswith('.npy')]
        
        # Pre-calculate sample indices
        self.samples = []
        for file_idx in tqdm(range(len(self.file_names))):
            # Load traffic context to determine number of frames
            traj_id_path = os.path.join(data_dir, 'traj_id_label', self.file_names[file_idx])
            traj_id = np.load(traj_id_path)
            
            # Calculate number of valid starting frames
            max_start = time_span - (input_frames + output_frames) + 1
            
            # Add samples for different starting frames
            for start_frame in range(0, max_start, stride):
                self.samples.append((file_idx, start_frame))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_idx, start_frame = self.samples[idx]
        file_name = self.file_names[file_idx]
        
        # Load traffic context
        traj_id_path = os.path.join(self.data_dir, 'traj_id_label', file_name)
        traj_id = np.load(traj_id_path)
        
        # Extract relevant time window
        end_frame = start_frame + self.input_frames + self.output_frames
        traj_id_window = traj_id[:, start_frame:end_frame]
        
        # Split into input and target
        input_context = traj_id_window[:, :self.input_frames]
        target_context = traj_id_window[:, self.input_frames:]
        
        # Also load speed information if available
        try:
            speed_path = os.path.join(self.data_dir, 'speed_label', file_name)
            speed_label = np.load(speed_path)
            speed_window = speed_label[:, start_frame:end_frame]
            
            input_speed = speed_window[:, :self.input_frames]
            target_speed = speed_window[:, self.input_frames:]
        except:
            # If speed not available, create dummy data
            input_speed = np.zeros_like(input_context)
            target_speed = np.zeros_like(target_context)
        
        # Convert to PyTorch tensors
        input_context_tensor = torch.FloatTensor(input_context)
        target_context_tensor = torch.FloatTensor(target_context)
        input_speed_tensor = torch.FloatTensor(input_speed)
        target_speed_tensor = torch.FloatTensor(target_speed)
        
        # Combine everything into a single traffic context tensor
        # Shape: [lane_cells, input_frames + output_frames]
        full_context_tensor = torch.cat(
            [input_context_tensor, target_context_tensor], 
            dim=1
        )
        
        # Extract vehicle IDs
        vehicle_ids = torch.unique(full_context_tensor)
        vehicle_ids = vehicle_ids[vehicle_ids > 0]  # Exclude 0 (empty)
        
        return {
            'traffic_context': full_context_tensor,
            'input_context': input_context_tensor,
            'target_context': target_context_tensor,
            'input_speed': input_speed_tensor,
            'target_speed': target_speed_tensor,
            'vehicle_ids': vehicle_ids,
            'start_frame': start_frame
        }
    
def social_lstm_collate(batch):
    """
    Custom collate function for Social LSTM batches
    """
    # Stack traffic contexts
    traffic_contexts = torch.stack([item['traffic_context'] for item in batch])
    input_contexts = torch.stack([item['input_context'] for item in batch])
    target_contexts = torch.stack([item['target_context'] for item in batch])
    
    # Collect vehicle IDs for each sample
    all_vehicle_ids = [item['vehicle_ids'] for item in batch]
    start_frames = [item['start_frame'] for item in batch]
    
    # Stack speed tensors if available
    if 'input_speed' in batch[0]:
        input_speeds = torch.stack([item['input_speed'] for item in batch])
        target_speeds = torch.stack([item['target_speed'] for item in batch])
    else:
        input_speeds = None
        target_speeds = None
    
    return {
        'traffic_context': traffic_contexts,
        'input_context': input_contexts,
        'target_context': target_contexts,
        'input_speed': input_speeds,
        'target_speed': target_speeds,
        'vehicle_ids': all_vehicle_ids,
        'start_frame': start_frames
    }