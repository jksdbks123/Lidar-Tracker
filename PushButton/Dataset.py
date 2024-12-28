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
            frames.append(frame)
        cap.release()

        # Convert to tensor and apply transforms
        frames = np.array(frames)
        if self.transform:
            frames = self.transform(frames)

        return torch.tensor(frames, dtype=torch.float32), label, location
    
def create_data_loaders(train_dir, val_dir, test_dir, batch_size=8, transform=None):
    train_dataset = VideoDataset(train_dir, transform=transform)
    val_dataset = VideoDataset(val_dir, transform=transform)
    test_dataset = VideoDataset(test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader