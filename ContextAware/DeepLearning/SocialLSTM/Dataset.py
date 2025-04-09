import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SocialLSTMDataset(Dataset):
    def __init__(self, h5_path, input_frames=10, output_frames=12, transform=None):
        """
        Dataset for Social LSTM training
        
        Args:
            h5_path: Path to HDF5 file containing data
            input_frames: Number of input frames
            output_frames: Number of output frames to predict
            transform: Optional transforms to apply
        """
        self.h5_path = h5_path
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.transform = transform
        
        # Open HDF5 file for reading
        self.h5_file = h5py.File(h5_path, 'r')
        
        # Store useful information
        self.num_samples = self.h5_file['traffic_context'].shape[0]
        self.metadata = self.h5_file['metadata'][:]
        
        # Precompute indices of valid vehicle trajectories
        self.valid_trajectories = self._find_valid_trajectories()
        
    def _find_valid_trajectories(self):
        """Find all valid vehicle trajectories that have enough data for training"""
        valid_trajectories = []
        
        # For each sample
        for sample_idx in range(self.num_samples):
            # Get vehicle IDs for this sample
            vehicle_ids = self.h5_file['vehicle_id'][sample_idx]  # [lane_cells, total_frames]
            
            # Find unique vehicle IDs (excluding -1 which denotes empty cells)
            unique_ids = np.unique(vehicle_ids)
            unique_ids = unique_ids[unique_ids >= 0]
            
            for vehicle_id in unique_ids:
                # Check if this vehicle has data for all input frames
                valid = True
                
                # Check if vehicle appears in all input frames
                for t in range(self.input_frames):
                    if not np.any(vehicle_ids[:, t] == vehicle_id):
                        valid = False
                        break
                
                # Check if vehicle appears in at least first output frame
                if not np.any(vehicle_ids[:, self.input_frames] == vehicle_id):
                    valid = False
                
                if valid:
                    valid_trajectories.append((sample_idx, vehicle_id))
        
        return valid_trajectories
    
    def __len__(self):
        return len(self.valid_trajectories)
    
    def __getitem__(self, idx):
        # Get sample index and vehicle ID
        sample_idx, vehicle_id = self.valid_trajectories[idx]
        
        # Load data from HDF5 file
        vehicle_ids = self.h5_file['vehicle_id'][sample_idx]  # [lane_cells, total_frames]
        traffic_context = self.h5_file['traffic_context'][sample_idx]  # [lane_cells, total_frames]
        
        # Extract vehicle positions for input frames
        input_positions = np.zeros(self.input_frames, dtype=np.float32)
        for t in range(self.input_frames):
            # Find the front position (smallest index) where this vehicle appears
            positions = np.where(vehicle_ids[:, t] == vehicle_id)[0]
            if len(positions) > 0:
                input_positions[t] = np.min(positions)
                
        # Extract vehicle positions for output frames
        output_positions = np.zeros(self.output_frames, dtype=np.float32)
        output_masks = np.zeros(self.output_frames, dtype=np.float32)  # Mask for valid positions
        
        for t in range(self.output_frames):
            frame_idx = self.input_frames + t
            positions = np.where(vehicle_ids[:, frame_idx] == vehicle_id)[0]
            if len(positions) > 0:
                output_positions[t] = np.min(positions)
                output_masks[t] = 1.0  # Mark as valid, 1.0 if vehicle is present
                
        # Extract traffic context for input frames
        input_context = traffic_context[:, :self.input_frames].transpose(1, 0)  # [input_frames, lane_cells]
        
        # Convert to PyTorch tensors
        input_positions = torch.FloatTensor(input_positions)
        input_context = torch.FloatTensor(input_context)
        output_positions = torch.FloatTensor(output_positions)
        output_masks = torch.FloatTensor(output_masks)
        
        # Apply transforms if specified
        if self.transform:
            input_positions, input_context, output_positions, output_masks = self.transform(
                input_positions, input_context, output_positions, output_masks
            )
        """
        input_positions: [input_frames] : positions of the vehicle in the input frames
        input_context: [input_frames, lane_cells] : traffic context for input frames
        output_positions: [output_frames] : positions of the vehicle in the output frames
        output_masks: [output_frames] : 1.0 if vehicle is present, 0.0 if not

        """

        return input_positions, input_context, output_positions, output_masks
    
    def close(self):
        """Close the HDF5 file"""
        self.h5_file.close()
        
    def __del__(self):
        """Close the HDF5 file when the dataset is deleted"""
        self.close()