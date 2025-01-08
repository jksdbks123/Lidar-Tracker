# Torch Dataset Class
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import torch
import albumentations as A


def extract_optical_flow(frames):

    prev_frame = frames[0]
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    optical_flow_frames = []
# Loop through video frames
    for i in range(1,len(frames)):
        next_frame = frames[i]
        next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, next_gray, None, 
            0.5, 3, 15, 3, 5, 1.2, 0
        )
    # Compute magnitude and angle
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Normalize magnitude
        normalized_magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        # Encode angle into sine and cosine
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)
        # Stack the normalized magnitude, sin and cos angle
        optical_flow = np.stack([normalized_magnitude, sin_angle, cos_angle], axis=-1)
        # Save or append the result
        optical_flow_frames.append(optical_flow)
        # Update the previous frame and previous gray
        prev_gray = next_gray

    optical_flow_frames = np.array(optical_flow_frames)

    return optical_flow_frames

def compute_optical_flow(prev_gray, next_gray):
    flow = cv2.calcOpticalFlowFarneback(
    prev_gray, next_gray, None, 
    0.5, 3, 15, 3, 5, 1.2, 0
)
# Compute magnitude and angle
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
# Normalize magnitude
    normalized_magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
# Encode angle into sine and cosine
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)
# Stack the normalized magnitude, sin and cos angle
    optical_flow = np.stack([normalized_magnitude, sin_angle, cos_angle], axis=-1)
    return optical_flow

class VideoDataset(Dataset):
    def __init__(self, data_dir, preprocess=None, augmentation=None):
        self.data_dir = data_dir
        self.preprocess = preprocess
        self.augmentation = augmentation
        self.video_files = []
        self.labels = []
        self.locations = []
        
        for video_file in os.listdir(data_dir):
            if video_file.endswith(".mp4"):
                label = int(video_file.split("_")[0])
                location = video_file.split("_")[-1].split(".")[0]
                self.video_files.append(os.path.join(data_dir, video_file))
                self.labels.append(label)
                self.locations.append(location)

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]
        location = self.locations[idx]

        # Load video frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        frames = np.array(frames) # (seq_len, h, w, c)

        if frames.shape[0] < 30: # Pad with zeros if video is less than 6.1 seconds
            frames = np.concatenate([frames, np.zeros((30-frames.shape[0], frames.shape[1],frames.shape[2],frames.shape[3]), dtype=np.uint8)], axis=0)
        # Convert to tensor and apply transforms
        if self.augmentation:
            frames = [self.augmentation(image=frame)['image'] for frame in frames]
            frames = np.array(frames)
        if self.preprocess:
            frames = self.preprocess(frames)
        return frames, label, location
    
# Custom Transform for Normalization
def preprocessing(frames):
    """ Normalize frames (batch_size, seq_len, h, w, c) """
    # to tensor
    frames = torch.tensor(frames)
    frames = frames / 255.0  # Scale pixel values to [0, 1]
    mean = torch.tensor([0.485, 0.456, 0.406])  # Imagenet mean for RGB
    std = torch.tensor([0.229, 0.224, 0.225])  # Imagenet std for RGB
    frames = (frames - mean) / std  # Normalize
    frames = frames.to(torch.float32).permute(0,3,1,2)
    # frames = torch.nn.functional.interpolate(frames.permute(0,3,1,2), size=224, mode = 'bilinear', align_corners=False)
    return frames

transform_aug = A.Compose([
    A.Illumination(p=0.5),
    A.Equalize(p=0.5),
    A.RandomSunFlare(p=0.5,flare_roi=(0,0,1,0.5)),
    A.ElasticTransform(p=0.3,alpha=1,sigma=50),
])

def create_data_loaders(train_dir, val_dir, batch_size=4, preprocess=None, augmentation=None):
    train_dataset = VideoDataset(train_dir, preprocess=preprocess, augmentation=augmentation)
    val_dataset = VideoDataset(val_dir, preprocess=preprocess)
    # test_dataset = VideoDataset(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader