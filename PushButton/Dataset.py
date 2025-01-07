# Torch Dataset Class
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import torch
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation,adjust_contrast,adjust_saturation,adjust_hue,hflip


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

class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=None, augmentation_dict=None):
        self.data_dir = data_dir
        self.transform = transform
        self.video_files = []
        self.labels = []
        self.locations = []
        self.augmentation_dict = augmentation_dict
        for video_file in os.listdir(data_dir):
            if video_file.endswith(".mp4"):
                label = int(video_file.split("_")[0])
                location = video_file.split("_")[-1].split(".")[0]
                self.video_files.append(os.path.join(data_dir, video_file))
                self.labels.append(label)
                self.locations.append(location)
        pt1 = (450, 750)
        pt2 = (750, 950)
        x,y,w,h = pt1[0],pt1[1],pt2[0]-pt1[0],pt2[1]-pt1[1]
        self.x = x
        self.y = y
        self.w = w
        self.h = h

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
            frame = frame[self.y:self.y+self.h, self.x:self.x+self.w]
            frames.append(frame)
        cap.release()
        frames = np.array(frames) # (seq_len, h, w, c)

        if frames.shape[0] < 61: # Pad with zeros if video is less than 6.1 seconds
            frames = np.concatenate([frames, np.zeros((61-frames.shape[0], frames.shape[1],frames.shape[2],frames.shape[3]), dtype=np.uint8)], axis=0)
        optical_flow_frames = extract_optical_flow(frames)

        frames = frames[20:55]
        optical_flow_frames = optical_flow_frames[20:55]
        # Convert to tensor and apply transforms
        if self.transform:
            frames = self.transform(frames)
            # normalize optical flow
            optical_flow_frames = torch.tensor(optical_flow_frames)
            optical_flow_frames = optical_flow_frames.permute(0,3,1,2)
            optical_flow_frames = optical_flow_frames.to(torch.float32)

        if self.augmentation_dict:
            brightness_factor = self.augmentation_dict.get("brightness", 0)
            # contrast_factor = self.augmentation_dict.get("contrast", 0)
            # saturation_factor = self.augmentation_dict.get("saturation", 0)
            # hue_factor = self.augmentation_dict.get("hue", 0)
            # h_flip_factor = self.augmentation_dict.get("h_flip", 0)
            # noise_factor = self.augmentation_dict.get("noise", 0)
            frames = adjust_brightness(frames, np.random.uniform(1-brightness_factor, 1+brightness_factor))
            # frames = adjust_contrast(frames, np.random.uniform(1-contrast_factor, 1+contrast_factor))
            # frames = adjust_saturation(frames, np.random.uniform(1-saturation_factor, 1+saturation_factor))
            # frames = adjust_hue(frames, np.random.uniform(-hue_factor, hue_factor))
            # if np.random.uniform() < h_flip_factor:
            #     frames = hflip(frames)
            # frames = frames + np.random.normal(0, noise_factor, frames.shape)
            # to float32
        
        return frames,optical_flow_frames, label, location
    
# Custom Transform for Normalization
def custom_transform(frames):
    """ Normalize frames (batch_size, seq_len, h, w, c) """
    # to tensor
    frames = torch.tensor(frames)
    frames = frames / 255.0  # Scale pixel values to [0, 1]
    mean = torch.tensor([0.485, 0.456, 0.406])  # Imagenet mean for RGB
    std = torch.tensor([0.229, 0.224, 0.225])  # Imagenet std for RGB
    frames = (frames - mean) / std  # Normalize
    frames = frames.to(torch.float32).permute(0,3,1,2)
    # # resize to 224x224
    # frames = torch.nn.functional.interpolate(frames.permute(0,3,1,2), size=224, mode = 'bilinear', align_corners=False)
    return frames

def create_data_loaders(train_dir, val_dir, batch_size=8, transform=None, augmentation_dict=None):
    train_dataset = VideoDataset(train_dir, transform=transform, augmentation_dict=augmentation_dict)
    val_dataset = VideoDataset(val_dir, transform=transform)
    # test_dataset = VideoDataset(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader