from Model import CNNLSTMAttention
from Dataset import create_data_loaders,transform_aug,preprocessing
import os
import torch
from tqdm import tqdm
import torch.optim as optim
from Loss import *
import time
from sklearn.metrics import confusion_matrix
import numpy as np
import json


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)

def calculate_metrics(y_true, y_pred,threshold=0.5):
    # Calculate precision, recall, f1
    y_pred_thresholded = (y_pred > threshold).astype(int)
    conf_matrix = confusion_matrix(y_true, y_pred_thresholded)
    precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    recall = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0])
    F1 = 2 * precision * recall / (precision + recall)
    return precision, recall, F1

def train_model(device,num_epochs,learning_rate,batch_size,criterion,transform_aug,preprocessing,train_folder,val_folder, run_dir):
    train_loader, val_loader = create_data_loaders(train_folder, val_folder, batch_size=batch_size, preprocess=preprocessing, augmentation=transform_aug)    # model = ResNetLSTM().to(device)
    # model = ResNetLSTMWithAttention().to(device)
    model = CNNLSTMAttention().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # early_stopping = EarlyStopping(patience=5, verbose=True)
    training_curves = {"train": [], "val": []}
    os.makedirs(run_dir, exist_ok=True)
    # create model save directory
    model_save_dir = os.path.join(run_dir, "model")
    os.makedirs(model_save_dir, exist_ok=True)
    # create training curves save directory
    curves_save_dir = os.path.join(run_dir, "curves")
    os.makedirs(curves_save_dir, exist_ok=True)
    # save training details, and name of criterion, model, optimizer, scheduler
    training_details = {
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "criterion": criterion.__class__.__name__,
        "model": model.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "scheduler": scheduler.__class__.__name__,
    }
    training_details_path = os.path.join(run_dir, "training_details.json")
    with open(training_details_path, "w") as f:
        json.dump(training_details,
                    f,
                    indent=4)
    train_curve_path = os.path.join(curves_save_dir, "training_curves.pth")
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}")
        with tqdm(total=len(train_loader), desc="Training") as pbar:
            for inputs, labels, _ in train_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()

                optimizer.zero_grad()
                outputs,attention_weights = model(inputs)
                # outputs = model(inputs)
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
        y_true = []
        y_pred = []
        with torch.no_grad():
            with tqdm(total=len(val_loader), desc="Validation") as pbar:
                for inputs, labels, _ in val_loader:

                    inputs, labels = inputs.to(device), labels.to(device).float()
                    outputs,attention_weights = model(inputs)
                    # outputs = model(inputs)
                    outputs = torch.flatten(outputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    training_curves["val"].append(loss.item())
                    # Calculate precision, recall, f1
                    y_true.extend(labels.cpu().numpy())
                    y_pred.extend(outputs.cpu().numpy())
                    pbar.set_postfix({"Batch Loss": loss.item()})
                    pbar.update(1)

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        precision, recall, F1 = calculate_metrics(y_true, y_pred)
        print(f"Precision: {precision}, Recall: {recall}, F1: {F1}")
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1} Train Loss: {train_loss}, Val Loss: {val_loss}")
        torch.save(training_curves, train_curve_path)
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
    
    

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # criterion = nn.BCELoss()
    criterion = FocalLoss()
    print(criterion)
    num_epochs=50
    learning_rate=0.001
    batch_size = 4
    run_dir = r"D:\LiDAR_Data\2ndPHB\Video\left_signal_0108_focal"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    train_folder = r'D:\LiDAR_Data\2ndPHB\Video\Dataset\L_signal\train'
    val_folder = r'D:\LiDAR_Data\2ndPHB\Video\Dataset\L_signal\val'
    train_model(device,num_epochs,learning_rate,batch_size,criterion,transform_aug,preprocessing,train_folder, val_folder, run_dir)