import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
class LayerInspector:
    def __init__(self):
        self.layers = OrderedDict()

    def hook_fn(self, module, input, output):
        class_name = module.__class__.__name__
        layer_name = f"{class_name}-{len(self.layers)}"
        self.layers[layer_name] = {
            'input_shape': tuple(input[0].shape),
            'output_shape': tuple(output.shape)
        }

    def print_layer_info(self):
        print("Layer Details:")
        for name, info in self.layers.items():
            print(f"{name}:")
            print(f"  Input shape: {info['input_shape']}")
            print(f"  Output shape: {info['output_shape']}")
            print()

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNetTrajectory(nn.Module):
    def __init__(self, n_channels=1):
        super(UNetTrajectory, self).__init__()
        
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, 2, kernel_size=1)

        self.inspector = LayerInspector()
        self._register_hooks()

    def _register_hooks(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.MaxPool2d, DoubleConv)):
                module.register_forward_hook(self.inspector.hook_fn)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        
        x = self.outc(x)
        
        speed, confidence = torch.split(x, 1, dim=1)
        speed = F.relu(speed)
        confidence = torch.sigmoid(confidence)
        
        return torch.cat([speed, confidence], dim=1)

    def print_layer_info(self):
        self.inspector.print_layer_info()

class TrajectoryDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.images = [f for f in os.listdir(image_dir) if f.endswith('.png')]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)

        # Load image and label
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = Image.open(label_path).convert('L')

        # Apply transformations
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        # Normalize label from [0, 255] to [0, 30] m/s
        # label = label / 255.0 * 30.0

        return image, label
# Assuming the UNetTrajectory model is defined as before

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    diff = torch.abs(pred - target)
    loss = (alpha * (1 - torch.exp(-diff)) ** gamma) * diff
    return loss.mean()

def custom_loss(pred, target, confidence):
    speed_loss = focal_loss(pred, target)
    # confidence_loss = -torch.mean(confidence * torch.log(confidence + 1e-8))
    return speed_loss

def train_model(model, train_loader, val_loader, num_epochs=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device {device}")
    model.to(device)
    saving_path = r'D:\TimeSpaceDiagramDataset\model\weights'
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            pred_speed, pred_conf = outputs[:,0:1,:,:], outputs[:,1:2,:,:]
            loss = custom_loss(pred_speed, targets, pred_conf)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                pred_speed, pred_conf = outputs[:,0:1,:,:], outputs[:,1:2,:,:]
                loss = custom_loss(pred_speed, targets, pred_conf)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(saving_path, 'best_model.pth'))
            print("Model saved! at val loss: ", best_val_loss)


    print("Training completed!")

if __name__ == "__main__":
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create datasets
    train_dataset = TrajectoryDataset(r'D:\TimeSpaceDiagramDataset\train\images', r'D:\TimeSpaceDiagramDataset\train\labels', transform=transform)
    val_dataset = TrajectoryDataset(r'D:\TimeSpaceDiagramDataset\val\images', r'D:\TimeSpaceDiagramDataset\val\labels', transform=transform)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8)
    # Create model and train
    model = UNetTrajectory(n_channels=1)
    train_model(model, train_dataloader, val_dataloader)