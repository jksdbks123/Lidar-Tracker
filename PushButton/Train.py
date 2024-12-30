from Model import ResNetLSTM,ResNetLSTMWithAttention
from Dataset import create_data_loaders,custom_transform
import os
import torch
from tqdm import tqdm
import torch.optim as optim
from Loss import *
import time

def train_model(device,num_epochs,learning_rate,batch_size,criterion,augmentation_dict,train_folder, val_folder, run_dir):
    train_loader, val_loader = create_data_loaders(train_folder, val_folder, batch_size=batch_size, transform=custom_transform, augmentation_dict=augmentation_dict)
    # model = ResNetLSTM().to(device)
    model = ResNetLSTMWithAttention().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    training_curves = {"train": [], "val": []}
    os.makedirs(run_dir, exist_ok=True)
    # create model save directory
    model_save_dir = os.path.join(run_dir, "model")
    os.makedirs(model_save_dir, exist_ok=True)
    # create training curves save directory
    curves_save_dir = os.path.join(run_dir, "curves")
    os.makedirs(curves_save_dir, exist_ok=True)
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}")
        with tqdm(total=len(train_loader), desc="Training") as pbar:
            for inputs, labels, _ in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()

                optimizer.zero_grad()
                outputs, attention_weights = model(inputs)
                outputs = torch.flatten(outputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                training_curves["train"].append(loss.item())
                pbar.set_postfix({"Batch Loss": loss.item()})
                pbar.update(1)

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc="Validation") as pbar:
                for inputs, labels, _ in val_loader:
                    
                    inputs, labels = inputs.to(device), labels.to(device).float()
                    outputs, attention_weights = model(inputs)
                    outputs = torch.flatten(outputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    training_curves["val"].append(loss.item())
                    pbar.set_postfix({"Batch Loss": loss.item()})
                    pbar.update(1)

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1} Train Loss: {train_loss}, Val Loss: {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save best model
            cur_time = time.strftime("%Y%m%d-%H%M%S")
            print(f"Saving best model at {cur_time}")
            best_model_path = os.path.join(model_save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
        # save current model
        current_model_path = os.path.join(model_save_dir, f"last_model.pth")
        torch.save(model.state_dict(), current_model_path)
    # Save training curves
    train_curve_path = os.path.join(curves_save_dir, "training_curves.pth")
    torch.save(training_curves, train_curve_path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # criterion = nn.BCELoss()
    criterion = FocalLoss()
    num_epochs=50
    learning_rate=0.0001
    batch_size = 2
    run_dir = r"D:\LiDAR_Data\2ndPHB\Video\run"
    train_folder = r'D:\LiDAR_Data\2ndPHB\Video\Dataset\train'
    val_folder = r'D:\LiDAR_Data\2ndPHB\Video\Dataset\val'
    augmentation_dict = {"brightness": 0.5, "contrast": 0.5, "saturation": 0.5, "hue": 0.5,'h_flip':0.5, 'noise': 0.2}
    train_model(device,num_epochs,learning_rate,batch_size,criterion,augmentation_dict,train_folder, val_folder, run_dir)