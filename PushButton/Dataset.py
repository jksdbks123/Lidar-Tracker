# Torch Dataset Class
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import torch

class VideoDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.video_files = []
        self.labels = []
        self.locations = []
        self.left_box = {"xmin": 200, "ymin": 700, "xmax": 900, "ymax": 1000} # Region of interest for the ped button in left side of the screen
        self.right_box = {"xmin": 1200, "ymin": 700, "xmax": 1900, "ymax": 1000} # Region of interest for the ped button in right side of the screen
        
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
            if location == "L":
                frame = frame[self.left_box["ymin"]:self.left_box["ymax"], self.left_box["xmin"]:self.left_box["xmax"]]
            else:
                frame = frame[self.right_box["ymin"]:self.right_box["ymax"], self.right_box["xmin"]:self.right_box["xmax"]]
            # resize frame to 224x224
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        cap.release()
        frames = np.array(frames) # (seq_len, h, w, c)
        # Convert to tensor and apply transforms
        if self.transform:
            frames = self.transform(frames)
        return frames, label, location
    
# Custom Transform for Normalization
def custom_transform(frames):
    """ Normalize frames (batch_size, seq_len, h, w, c) """
    # to tensor
    frames = torch.tensor(frames)
    frames = frames / 255.0  # Scale pixel values to [0, 1]
    mean = torch.tensor([0.485, 0.456, 0.406])  # Imagenet mean for RGB
    std = torch.tensor([0.229, 0.224, 0.225])  # Imagenet std for RGB
    frames = (frames - mean) / std  # Normalize

    return frames.permute(0,3,1,2) # (seq_len, c, h, w)

def create_data_loaders(train_dir, val_dir, batch_size=8, transform=None):
    train_dataset = VideoDataset(train_dir, transform=transform)
    val_dataset = VideoDataset(val_dir, transform=transform)
    # test_dataset = VideoDataset(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader