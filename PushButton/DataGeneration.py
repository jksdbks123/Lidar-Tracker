import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from Dataset import *
from torchvision import transforms
from p_tqdm import p_umap
from functools import partial

frame_width,frame_height = 250,150
x,y = 450,780
pt1_L = (x, y)
pt2_L = (450+frame_width, 780+frame_height)
x,y = 1450,800
pt1_R = (x, y)
pt2_R = (x+frame_width, y+frame_height)
ROI_L, ROI_R = [pt1_L, pt2_L], [pt1_R, pt2_R]
def write_video(output_path, frames_L,frames_R, fourcc, fps, frame_width, frame_height,location):
    """
    Writes a video file using OpenCV.

    Args:
        output_path (str): Path to save the video.
        frames (list): List of frames to write.
        fourcc (int): Codec.
        fps (int): Frames per second.
        frame_width (int): Width of the frame.
        frame_height (int): Height of the frame.
    """
    # Set up VideoWriter for saving the output
    if os.path.exists(output_path):
       return None
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if location == 'L':
        for frame in frames_L:
            out.write(frame)
    else:
        for frame in frames_R:
            out.write(frame)
    # Release resources
    out.release()

def clip_single_video(input_video_path, save_names, target_frames, locations, output_folder):
    """
    Clips a video using OpenCV based on start and end timestamps.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the clipped video.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
    """
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

    init_frame = target_frames[:,0].min()
    ending_frame = target_frames[:,1].max()
    current_frame = init_frame
    # Set the starting position of the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)

    frames_L = []
    frames_R = []
    frame_inds = []
    while current_frame <= ending_frame:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached or error reading frame.")
            break
        frame_L = frame[ROI_L[0][1]:ROI_L[1][1], ROI_L[0][0]:ROI_L[1][0]].copy()
        frame_R = frame[ROI_R[0][1]:ROI_R[1][1], ROI_R[0][0]:ROI_R[1][0]].copy()
        del frame
        frames_L.append(frame_L)
        frames_R.append(frame_R)
        frame_inds.append(current_frame)
        current_frame += 1
    frame_inds = np.array(frame_inds)
    if len(frame_inds) == 0:
        return
    for i in range(len(target_frames)):
        start_frame = target_frames[i,0]
        end_frame = target_frames[i,1]
        try:
            start_ind = np.where(frame_inds == start_frame)[0][0]
            end_ind = np.where(frame_inds == end_frame)[0][0]
        except:
            continue
        save_name = save_names[i]
        location = locations[i]
        output_path = os.path.join(output_folder, save_name)
        write_video(output_path, frames_L[start_ind:end_ind], frames_L[start_ind:end_ind],fourcc, fps, frame_width, frame_height,location)
    # Release resources
    cap.release()

# Load Excel file
def load_excel_data(file_path):
    activation_df = pd.read_excel(file_path, sheet_name="Activation")
    non_activation_df = pd.read_excel(file_path, sheet_name="NonActivation")
    return activation_df, non_activation_df

def generate_frame_list(video_dir, time_window, activation_df, non_activation_df, student_file):
    """
Conbine the my labels and high school student labels
"""
    student_activation_L = student_file.loc[(student_file.Bound == 'N') & (student_file.loc[:,'Pressed?'] == 'Y')].copy()
    student_activation_R = student_file.loc[(student_file.Bound == 'S') & (student_file.loc[:,'Pressed?'] == 'Y')].copy()
# uniform the column names and data format
# combine Date column and Time column in student_activation_L and student_activation_R, and convert to YYYYMMDD_HHMMSS format
    timestamp = student_activation_L['Date'].astype(str) + '_' + student_activation_L['Time'].astype(str)
    timestamp = timestamp.str.replace(':','')
# eliminate '-' in the timestamp
    timestamp = timestamp.str.replace('-','')
    student_activation_L.loc[:,'timestamp'] = timestamp
    timestamp = student_activation_R['Date'].astype(str) + '_' + student_activation_R['Time'].astype(str)
    timestamp = timestamp.str.replace(':','')
    timestamp = timestamp.str.replace('-','')
    student_activation_R.loc[:,'timestamp'] = timestamp
# combine student_activation_L and student_activation_R to activation_df
    student_activation_L.loc[:, 'location'] = 'L'
    student_activation_R.loc[:, 'location'] = 'R'
    student_activation_L = student_activation_L.loc[:,['timestamp','location']]
    student_activation_R = student_activation_R.loc[:,['timestamp','location']]
    student_activation = pd.concat([student_activation_L, student_activation_R], axis=0)
    activation_df = pd.concat([activation_df, student_activation], axis=0)
# filter duplicate rows
    activation_df.drop_duplicates(inplace=True)
    """
Generate clipping list for the dataset
"""
# Prepare data for activation and non-activation
    date_times = []
    start_frames = []
    end_frames = []
    save_names = []
    for df, label in [(activation_df, 1), (non_activation_df, 0)]:
        for _, row in tqdm(df.iterrows()):
            record_timestamp = row["timestamp"]
            location = row["location"]
        # convert to datetime object
            record_timestamp = datetime.strptime(record_timestamp, "%Y%m%d_%H%M%S")
        
        # Match the video file with the timestamp
            for video_file in os.listdir(video_dir):
                video_start_timestamp = video_file[22:].split(".")[0]
            # convert to datetime object
                video_start_timestamp = datetime.strptime(video_start_timestamp, "%Y%m%d_%H%M%S")
                video_end_timestamp = video_start_timestamp + timedelta(seconds=60 * 5)
                if video_start_timestamp <= record_timestamp <= video_end_timestamp:
                # convert video_start_timestamp to '%Y-%m-%d-%H-%M-%S'
                    video_start_timestamp_str = video_start_timestamp.strftime('%Y-%m-%d-%H-%M-%S')
                # window screening for the video clips making sure the record_timestamp is each 30 seconds
                    screen_start_frame = int((record_timestamp - video_start_timestamp).seconds * fps)
                    screen_end_frame = screen_start_frame + time_window * fps
                    for step in range(2,18): # 2 frame margin for consevative screening
                        start_frame = int(screen_start_frame + step)
                        end_frame = int(screen_end_frame + step)
                        start_frames.append(start_frame)
                        end_frames.append(end_frame)
                        date_times.append(video_start_timestamp_str)
                        save_names.append(f"{label}_{video_start_timestamp_str}_{start_frame}_{location}.mp4")

# make the dataframe
    data = {"date_time": date_times, "start_frame": start_frames, "end_frame": end_frames, "save_name": save_names}
    df = pd.DataFrame(data)
# video file format: 00_00_192.168.1.108_1_20241204_224500.avi
    target_video_paths = []
    target_frames = []
    save_names = []
    locations = []
    for date_time,g in df.groupby("date_time"):
    # convert date_time to YYYYMMDD_HHMMSS
        date_time = datetime.strptime(date_time, "%Y-%m-%d-%H-%M-%S").strftime("%Y%m%d_%H%M%S")
        video_file = f"00_00_192.168.1.108_1_{date_time}.avi"
        video_path = os.path.join(video_dir, video_file)
        target_video_paths.append(video_path)
        target_start_frames = g["start_frame"].tolist()
        target_end_frames = g["end_frame"].tolist()
        target_frames.append(np.c_[target_start_frames, target_end_frames])
        save_names.append(g["save_name"].tolist())
        locations.append(g["save_name"].str[-5].tolist())
    return save_names,target_video_paths,target_frames,locations


if __name__ == "__main__":
    n_cpu = 6
    # Process the dataset
    video_dir = r"D:\LiDAR_Data\2ndPHB\Video\IntesectionVideo"
    excel_file_path = r"D:\LiDAR_Data\2ndPHB\Video\Activation.xlsx"
    output_dir = r"D:\LiDAR_Data\2ndPHB\Video\Dataset"
    clip_save_dir = r"D:\LiDAR_Data\2ndPHB\Video\Clips"
    if not os.path.exists(clip_save_dir):
        os.makedirs(clip_save_dir)
    time_window = 3 # seconds
    # Load the Excel data
    activation_df, non_activation_df = load_excel_data(excel_file_path)
    excel_file_1 = r"D:\LiDAR_Data\2ndPHB\Video\HAWK Pedestrian Behavior.xlsx"
    student_file = pd.read_excel(excel_file_1)
    time_window = 3 # seconds (30 frames)
    fps = 10
    save_names, target_video_paths, target_frames, locations = generate_frame_list(video_dir, time_window, activation_df, non_activation_df, student_file)

    p_umap(
            partial(clip_single_video, output_folder=clip_save_dir),
            target_video_paths,
            save_names,
            target_frames,
            locations,
            num_cpus = n_cpu
        )