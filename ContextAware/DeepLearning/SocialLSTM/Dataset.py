import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import os

class TrajectoryDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        Dataset for trajectory data with position and neighbor information
        
        Args:
            data_path: Path to HDF5 file or directory containing data files
            transform: Optional transform to apply to the data
        """
        self.data_path = data_path
        self.transform = transform
        self.total_samples = 0
        self.sample_map = []  # Maps index -> (file_path, batch_idx, mini_batch_idx)
        
        # Check if data_path is a file or directory
        if os.path.isfile(data_path) and data_path.endswith('.h5'):
            # Single H5 file
            self._index_h5_file(data_path)
        elif os.path.isdir(data_path):
            # Directory of files
            self._index_directory(data_path)
        else:
            raise ValueError(f"Invalid data path: {data_path}. Must be an H5 file or directory.")

    def _index_h5_file(self, file_path):
        """Index all samples in an H5 file for efficient access"""
        with h5py.File(file_path, 'r') as f:
            # Get total trajectory count if available as attribute
            if 'total_trajectory_samples' in f.attrs:
                self.total_samples = f.attrs['total_trajectory_samples']
            
            # Iterate through batch groups
            for batch_name in f.keys():
                batch_group = f[batch_name]
                
                # Iterate through mini-batches
                for mini_batch_name in batch_group.keys():
                    mini_batch = batch_group[mini_batch_name]
                    
                    # Get batch size from the inputs dataset shape
                    if 'inputs' in mini_batch:
                        batch_size = mini_batch['inputs'].shape[0]
                        for i in range(batch_size):
                            self.sample_map.append((file_path, batch_name, mini_batch_name, i))
                    
            # Update total_samples if we didn't get it from attributes
            if self.total_samples == 0:
                self.total_samples = len(self.sample_map)

    def _index_directory(self, dir_path):
        """Index all H5 files in a directory"""
        for file_name in os.listdir(dir_path):
            if file_name.endswith('.h5'):
                file_path = os.path.join(dir_path, file_name)
                self._index_h5_file(file_path)

    def __len__(self):
        return len(self.sample_map)

    def __getitem__(self, idx):
        """Get a single trajectory sample"""
        if idx >= len(self.sample_map):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.sample_map)} samples")
        
        # Get file and location information
        file_path, batch_name, mini_batch_name, sample_idx = self.sample_map[idx]
        
        # Open the file and retrieve sample
        with h5py.File(file_path, 'r') as f:
            mini_batch = f[batch_name][mini_batch_name]
            inputs = torch.from_numpy(mini_batch['inputs'][sample_idx].astype(np.float32))
            targets = torch.from_numpy(mini_batch['targets'][sample_idx].astype(np.float32))
        
        # Apply optional transforms
        if self.transform:
            inputs, targets = self.transform(inputs, targets)
        
        return {'inputs': inputs, 'targets': targets}
    
    def get_batch(self, file_path, batch_name, mini_batch_name):
        """Get an entire mini-batch directly (more efficient than individual samples)"""
        with h5py.File(file_path, 'r') as f:
            mini_batch = f[batch_name][mini_batch_name]
            inputs = torch.from_numpy(mini_batch['inputs'][:].astype(np.float32))
            targets = torch.from_numpy(mini_batch['targets'][:].astype(np.float32))
        
        return {'inputs': inputs, 'targets': targets}