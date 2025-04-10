import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import h5py

# class SocialLSTMDataset(Dataset):
#     def __init__(self, h5_path, input_frames=10, output_frames=12, lane_cells=200, sigma=2.0, transform=None, cache_size=1000):
#         """
#         Dataset for Social LSTM training with distribution targets that supports pickling
        
#         Args:
#             h5_path: Path to HDF5 file containing data
#             input_frames: Number of input frames
#             output_frames: Number of output frames to predict
#             lane_cells: Number of lane cells
#             sigma: Standard deviation for Gaussian distribution around target position
#             transform: Optional transforms to apply
#             cache_size: Maximum number of samples to cache in memory
#         """
#         self.h5_path = h5_path
#         self.input_frames = input_frames
#         self.output_frames = output_frames
#         self.lane_cells = lane_cells
#         self.sigma = sigma
#         self.transform = transform
#         self.cache_size = cache_size
        
#         # Initialize cache
#         self.cache = {}
        
#         # Perform initial file read to get necessary metadata
#         self._initialize()
    
#     def _initialize(self):
#         """Initialize dataset by reading metadata once"""
#         with h5py.File(self.h5_path, 'r') as h5_file:
#             # Store useful information
#             self.num_samples = h5_file['traffic_context'].shape[0]
            
#             # Read metadata
#             self.metadata = h5_file['metadata'][:]
            
#             # Find valid trajectories (store indices only, not data)
#             self.valid_trajectories = self._find_valid_trajectories(h5_file)
    
#     def _find_valid_trajectories(self, h5_file):
#         """Find all valid vehicle trajectories that have enough data for training"""
#         valid_trajectories = []
        
#         # For each sample
#         for sample_idx in range(min(self.num_samples, 1000)):  # Limit initial scan for large datasets
#             # Get vehicle IDs for this sample
#             vehicle_ids = h5_file['vehicle_id'][sample_idx]  # [lane_cells, total_frames]
            
#             # Find unique vehicle IDs (excluding -1 which denotes empty cells)
#             unique_ids = np.unique(vehicle_ids)
#             unique_ids = unique_ids[unique_ids >= 0]
            
#             for vehicle_id in unique_ids:
#                 # Check if this vehicle has data for all input frames
#                 valid = True
                
#                 # Check if vehicle appears in all input frames
#                 for t in range(self.input_frames):
#                     if not np.any(vehicle_ids[:, t] == vehicle_id):
#                         valid = False
#                         break
                
#                 # Check if vehicle appears in at least first output frame
#                 if not np.any(vehicle_ids[:, self.input_frames] == vehicle_id):
#                     valid = False
                
#                 if valid:
#                     valid_trajectories.append((sample_idx, int(vehicle_id)))
        
#         return valid_trajectories
    
#     def _create_gaussian_distribution(self, position, sigma=None):
#         """
#         Create a Gaussian distribution centered at position
        
#         Args:
#             position: Center position for the Gaussian
#             sigma: Standard deviation (if None, use self.sigma)
            
#         Returns:
#             distribution: Gaussian distribution across lane cells
#         """
#         if sigma is None:
#             sigma = self.sigma
            
#         # Create array for all lane cell positions
#         cell_positions = np.arange(self.lane_cells)
        
#         # Create Gaussian distribution
#         distribution = np.exp(-0.5 * ((cell_positions - position) / sigma) ** 2)
        
#         # Normalize to sum to 1
#         distribution = distribution / np.sum(distribution)
        
#         return distribution
    
#     def __len__(self):
#         return len(self.valid_trajectories)
    
#     def __getitem__(self, idx):
#         # Check if the data is in cache
#         if idx in self.cache:
#             return self.cache[idx]
        
#         # Get sample index and vehicle ID
#         sample_idx, vehicle_id = self.valid_trajectories[idx]
        
#         # Open file for reading
#         with h5py.File(self.h5_path, 'r') as h5_file:
#             # Load data from HDF5 file
#             vehicle_ids = h5_file['vehicle_id'][sample_idx]  # [lane_cells, total_frames]
#             traffic_context = h5_file['traffic_context'][sample_idx]  # [lane_cells, total_frames]
            
#             # Extract vehicle positions for input frames
#             input_positions = np.zeros(self.input_frames, dtype=np.float32)
#             for t in range(self.input_frames):
#                 # Find the front position (smallest index) where this vehicle appears
#                 positions = np.where(vehicle_ids[:, t] == vehicle_id)[0]
#                 if len(positions) > 0:
#                     input_positions[t] = np.min(positions)
                    
#             # Extract vehicle positions for output frames and create distributions
#             target_positions = np.zeros(self.output_frames, dtype=np.float32)
#             output_distributions = np.zeros((self.output_frames, self.lane_cells), dtype=np.float32)
#             output_masks = np.zeros(self.output_frames, dtype=np.float32)  # Mask for valid positions
            
#             for t in range(self.output_frames):
#                 frame_idx = self.input_frames + t
#                 positions = np.where(vehicle_ids[:, frame_idx] == vehicle_id)[0]
                
#                 if len(positions) > 0:
#                     position = np.min(positions)
#                     target_positions[t] = position
#                     output_distributions[t] = self._create_gaussian_distribution(position)
#                     output_masks[t] = 1.0  # Mark as valid
#                 else:
#                     # If vehicle is not present, create a uniform distribution
#                     output_distributions[t] = np.ones(self.lane_cells) / self.lane_cells
                    
#             # Extract traffic context for input frames
#             input_context = traffic_context[:, :self.input_frames].transpose(1, 0)  # [input_frames, lane_cells]
        
#         # Convert to PyTorch tensors
#         input_positions = torch.FloatTensor(input_positions)
#         input_context = torch.FloatTensor(input_context)
#         target_positions = torch.FloatTensor(target_positions)
#         output_distributions = torch.FloatTensor(output_distributions)
#         output_masks = torch.FloatTensor(output_masks)
        
#         # Apply transforms if specified
#         if self.transform:
#             input_positions, input_context, target_positions, output_distributions, output_masks = self.transform(
#                 input_positions, input_context, target_positions, output_distributions, output_masks
#             )
        
#         # Create sample tuple
#         sample = (input_positions, input_context, target_positions, output_distributions, output_masks)
        
#         # Cache the sample if cache is not full
#         if len(self.cache) < self.cache_size:
#             self.cache[idx] = sample
        
#         return sample
    
#     def __getstate__(self):
#         """
#         Return a pickleable state for this object
        
#         Removes the h5_file handle which is not pickleable
#         """
#         state = self.__dict__.copy()
#         # Remove the cache which may contain non-pickleable objects
#         state['cache'] = {}
#         return state
    
#     def __setstate__(self, state):
#         """
#         Restore the state of this object
        
#         Reopens the h5_file handle
#         """
#         self.__dict__.update(state)
#         # Reinitialize with the h5_path
#         self._initialize()

class SocialLSTMDataset(Dataset):
    def __init__(self, h5_path, input_frames=10, output_frames=12, lane_cells=200, sigma=2.0, transform=None):
        """
        Dataset for Social LSTM training with distribution targets
        
        Args:
            h5_path: Path to HDF5 file containing data
            input_frames: Number of input frames
            output_frames: Number of output frames to predict
            lane_cells: Number of lane cells
            sigma: Standard deviation for Gaussian distribution around target position
            transform: Optional transforms to apply
        """
        self.h5_path = h5_path
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.lane_cells = lane_cells
        self.sigma = sigma
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
    
    def _create_gaussian_distribution(self, position, sigma=None):
        """
        Create a Gaussian distribution centered at position
        
        Args:
            position: Center position for the Gaussian
            sigma: Standard deviation (if None, use self.sigma)
            
        Returns:
            distribution: Gaussian distribution across lane cells
        """
        if sigma is None:
            sigma = self.sigma
            
        # Create array for all lane cell positions
        cell_positions = np.arange(self.lane_cells)
        
        # Create Gaussian distribution
        distribution = np.exp(-0.5 * ((cell_positions - position) / sigma) ** 2)
        
        # Normalize to sum to 1
        distribution = distribution / np.sum(distribution)
        
        return distribution
    
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
                
        # Extract vehicle positions for output frames and create distributions
        target_positions = np.zeros(self.output_frames, dtype=np.float32)
        output_distributions = np.zeros((self.output_frames, self.lane_cells), dtype=np.float32)
        output_masks = np.zeros(self.output_frames, dtype=np.float32)  # Mask for valid positions
        
        for t in range(self.output_frames):
            frame_idx = self.input_frames + t
            positions = np.where(vehicle_ids[:, frame_idx] == vehicle_id)[0]
            
            if len(positions) > 0:
                position = np.min(positions)
                target_positions[t] = position
                output_distributions[t] = self._create_gaussian_distribution(position)
                output_masks[t] = 1.0  # Mark as valid
            else:
                # If vehicle is not present, create a uniform distribution
                output_distributions[t] = np.ones(self.lane_cells) / self.lane_cells
                
        # Extract traffic context for input frames
        input_context = traffic_context[:, :self.input_frames].transpose(1, 0)  # [input_frames, lane_cells]
        
        # Convert to PyTorch tensors
        input_positions = torch.FloatTensor(input_positions)
        input_context = torch.FloatTensor(input_context)
        target_positions = torch.FloatTensor(target_positions)
        output_distributions = torch.FloatTensor(output_distributions)
        output_masks = torch.FloatTensor(output_masks)
        
        # Apply transforms if specified
        if self.transform:
            input_positions, input_context, target_positions, output_distributions, output_masks = self.transform(
                input_positions, input_context, target_positions, output_distributions, output_masks
            )
        
        return input_positions, input_context, target_positions, output_distributions, output_masks
    
    def close(self):
        """Close the HDF5 file"""
        self.h5_file.close()
        
    def __del__(self):
        """Close the HDF5 file when the dataset is deleted"""
        self.close()

class MemoryMappedSocialLSTMDataset(Dataset):
    def __init__(self, h5_path, input_frames=10, output_frames=12, lane_cells=200, sigma=2.0,
                 preprocess_dir=None, transform=None):
        """
        Memory-mapped dataset for Social LSTM training
        
        This dataset creates memory-mapped files for faster access during training
        
        Args:
            h5_path: Path to HDF5 file containing data
            input_frames: Number of input frames
            output_frames: Number of output frames to predict
            lane_cells: Number of lane cells
            sigma: Standard deviation for Gaussian distribution around target position
            preprocess_dir: Directory to store preprocessed data (if None, use same dir as h5_path)
            transform: Optional transforms to apply
        """
        self.h5_path = h5_path
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.lane_cells = lane_cells
        self.sigma = sigma
        self.transform = transform
        
        # Set preprocessing directory
        if preprocess_dir is None:
            self.preprocess_dir = os.path.dirname(h5_path)
        else:
            self.preprocess_dir = preprocess_dir
            os.makedirs(preprocess_dir, exist_ok=True)
        
        # Determine cache file path
        h5_filename = os.path.basename(h5_path)
        self.cache_file = os.path.join(
            self.preprocess_dir, 
            f"{h5_filename}_preprocessed_i{input_frames}_o{output_frames}_s{sigma:.1f}.npz"
        )
        
        # Check if preprocessed data exists, otherwise create it
        if not os.path.exists(self.cache_file):
            self._preprocess_data()
        
        # Load preprocessed data
        self._load_preprocessed_data()
    
    def _create_gaussian_distribution(self, position, sigma=None):
        """Create a Gaussian distribution centered at position"""
        if sigma is None:
            sigma = self.sigma
            
        cell_positions = np.arange(self.lane_cells)
        distribution = np.exp(-0.5 * ((cell_positions - position) / sigma) ** 2)
        distribution = distribution / np.sum(distribution)
        
        return distribution
    
    def _preprocess_data(self):
        """Preprocess data and save to disk for memory mapping"""
        print(f"Preprocessing data from {self.h5_path}...")
        
        # Open HDF5 file for reading
        with h5py.File(self.h5_path, 'r') as h5_file:
            # Store useful information
            num_samples = h5_file['traffic_context'].shape[0]
            
            # Find valid trajectories
            valid_trajectories = []
            sample_indices = []
            vehicle_ids_list = []
            
            for sample_idx in tqdm(range(num_samples), desc="Finding valid trajectories"):
                # Get vehicle IDs for this sample
                vehicle_ids = h5_file['vehicle_id'][sample_idx]  # [lane_cells, total_frames]
                
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
                        sample_indices.append(sample_idx)
                        vehicle_ids_list.append(vehicle_id)
            
            # Allocate arrays for preprocessed data
            num_valid = len(valid_trajectories)
            input_positions = np.zeros((num_valid, self.input_frames), dtype=np.float32)
            input_context = np.zeros((num_valid, self.input_frames, self.lane_cells), dtype=np.float32)
            target_positions = np.zeros((num_valid, self.output_frames), dtype=np.float32)
            output_distributions = np.zeros((num_valid, self.output_frames, self.lane_cells), dtype=np.float32)
            output_masks = np.zeros((num_valid, self.output_frames), dtype=np.float32)
            
            # Process each valid trajectory
            for i, (sample_idx, vehicle_id) in enumerate(tqdm(valid_trajectories, desc="Processing trajectories")):
                # Get data for this sample
                vehicle_ids = h5_file['vehicle_id'][sample_idx]
                traffic_context = h5_file['traffic_context'][sample_idx]
                
                # Extract vehicle positions for input frames
                for t in range(self.input_frames):
                    positions = np.where(vehicle_ids[:, t] == vehicle_id)[0]
                    if len(positions) > 0:
                        input_positions[i, t] = np.min(positions)
                
                # Extract vehicle positions for output frames and create distributions
                for t in range(self.output_frames):
                    frame_idx = self.input_frames + t
                    positions = np.where(vehicle_ids[:, frame_idx] == vehicle_id)[0]
                    
                    if len(positions) > 0:
                        position = np.min(positions)
                        target_positions[i, t] = position
                        output_distributions[i, t] = self._create_gaussian_distribution(position)
                        output_masks[i, t] = 1.0
                    else:
                        output_distributions[i, t] = np.ones(self.lane_cells) / self.lane_cells
                
                # Extract traffic context for input frames
                input_context[i] = traffic_context[:, :self.input_frames].transpose(1, 0)
        
        # Save preprocessed data
        np.savez_compressed(
            self.cache_file,
            valid_trajectories=np.array(valid_trajectories),
            input_positions=input_positions,
            input_context=input_context,
            target_positions=target_positions,
            output_distributions=output_distributions,
            output_masks=output_masks
        )
        
        print(f"Preprocessed data saved to {self.cache_file}")
    
    def _load_preprocessed_data(self):
        """Load preprocessed data for memory mapping"""
        print(f"Loading preprocessed data from {self.cache_file}...")
        
        data = np.load(self.cache_file)
        self.valid_trajectories = data['valid_trajectories']
        self.input_positions = data['input_positions']
        self.input_context = data['input_context']
        self.target_positions = data['target_positions']
        self.output_distributions = data['output_distributions']
        self.output_masks = data['output_masks']
        
        print(f"Loaded {len(self.valid_trajectories)} valid trajectories")
    
    def __len__(self):
        return len(self.valid_trajectories)
    
    def __getitem__(self, idx):
        # Get data from memory-mapped arrays
        input_positions = torch.FloatTensor(self.input_positions[idx])
        input_context = torch.FloatTensor(self.input_context[idx])
        target_positions = torch.FloatTensor(self.target_positions[idx])
        output_distributions = torch.FloatTensor(self.output_distributions[idx])
        output_masks = torch.FloatTensor(self.output_masks[idx])
        
        # Apply transforms if specified
        if self.transform:
            input_positions, input_context, target_positions, output_distributions, output_masks = self.transform(
                input_positions, input_context, target_positions, output_distributions, output_masks
            )
        
        return input_positions, input_context, target_positions, output_distributions, output_masks