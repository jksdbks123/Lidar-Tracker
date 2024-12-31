import cv2
from p_tqdm import p_umap
from functools import partial
from threading import Thread
import os
import pandas as pd
import numpy as np

def write_video(output_path, frames, fourcc, fps, frame_width, frame_height):
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
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    for frame in frames:
        out.write(frame)
    # Release resources
    out.release()

def analyze_availability(video_folder,ref_table,date_column_name, frame_column_name, output_name_column, time_interval, fps = 10):
    # video name format: 00_00_192.168.1.108_1_20241208_231000.avi
    video_list = os.listdir(video_folder)
    video_list = [f for f in video_list if f.endswith('.avi')]
    if len(video_list) == 0:
        return None,None,None
    date_str = []
    for f in video_list:
        date = f[22:-4] # YYYYMMDD_HHMMSS
        date_str.append(date)
    video_date = pd.to_datetime(pd.Series(date_str),format=('%Y%m%d_%H%M%S'))
    query_date = pd.to_datetime(ref_table.loc[:,date_column_name],format=('%Y-%m-%d-%H-%M-%S')) 

    query_frame_index = ref_table.loc[:,frame_column_name]
    output_names = ref_table.loc[:,output_name_column]

    video_inds = []
    for i in range(len(query_date)):
        TimeDiff = (query_date.iloc[i] - video_date)
        within5 = (TimeDiff < pd.Timedelta(5,unit='Minute')) & ((TimeDiff >= pd.Timedelta(0,unit='Minute')))
        valid_video_ind = TimeDiff.loc[within5].argsort().index[0] if within5.sum() > 0 else -1
        video_inds.append(valid_video_ind)

    video_inds = np.array(video_inds)
    uni_inds = np.unique(video_inds)

    target_frames = []
    video_paths_ = []
    output_names_ = []

    for i in uni_inds:
        if i == -1:
            continue
        start_frames = np.array(query_frame_index.loc[video_inds==i] - time_interval * fps).reshape(-1,1)
        end_frames = np.array(query_frame_index.loc[video_inds==i] + time_interval * fps).reshape(-1,1)
        start_frames[start_frames < 0] = 0
        end_frames[end_frames > 2999] = 2999
        target_frames.append(np.concatenate([start_frames,end_frames],axis = 1))
        video_paths_.append(os.path.join(video_folder,video_list[i]))
        output_names_.append(output_names.loc[video_inds==i].values)

    return target_frames,video_paths_,output_names_

def clip_single_video(input_video_path, save_name,target_frames,output_folder):
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
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the frame
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

    init_frame = target_frames[:,0].min()
    ending_frame = target_frames[:,1].max()
    current_frame = init_frame
    # Set the starting position of the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, init_frame)

    frames = []
    frame_inds = []
    while current_frame <= ending_frame:
        ret, frame = cap.read()
        if not ret:
            # print("End of video reached or error reading frame.")
            break
        frames.append(frame)
        frame_inds.append(current_frame)
        current_frame += 1
    for i in range(len(target_frames)):
        start_frame = target_frames[i,0]
        end_frame = target_frames[i,1]
        output_path = os.path.join(output_folder,save_name[i] + '.avi')
        start_ind = np.where(frame_inds == start_frame)[0][0]
        end_ind = np.where(frame_inds == end_frame)[0][0]
        write_video(output_path, frames[start_ind:end_ind], fourcc, fps, frame_width, frame_height)
    # Release resources
    cap.release()

def batch_video_clipping(video_folder, output_folder, ref_table_path, date_column_name, frame_column_name, time_interval, output_name_column, n_cpu):
    """
    Batch clips videos in a folder based on start and end frames.

    Args:
        input_folder (str): Path to the folder containing input videos.
        output_folder (str): Path to save the clipped videos.
        start_frames (list): List of start frames for each video.
        end_frames (list): List of end frames for each video.
    """
    ref_table = pd.read_csv(ref_table_path)
    target_frames,video_paths_,output_names_ = analyze_availability(video_folder,ref_table,date_column_name, frame_column_name, output_name_column, time_interval, fps = 10)
    if target_frames is None:
        print('No video found')

    p_umap(
        partial(clip_single_video, output_folder=output_folder),
        video_paths_,
        output_names_,
        target_frames,
        num_cpus = n_cpu
    )
def run_batch_video_clipping_threaded(video_folder, output_folder, ref_table_path, date_column_name, frame_column_name, time_interval, output_name_column, n_cpu):
    """
    Runs the batch video clipping in a separate thread.

    Args:
        input_folder (str): Path to the folder containing input videos.
        output_folder (str): Path to save the clipped videos.
        start_frames (list): List of start frames for each video.
        end_frames (list): List of end frames for each video.
    """
    thread = Thread(
        target=batch_video_clipping,
        args=(video_folder, output_folder, ref_table_path, date_column_name, frame_column_name, time_interval, output_name_column, n_cpu),
    )
    thread.start()
    

if __name__ == "__main__":
    ref_table = pd.read_csv(r'D:\LiDAR_Data\2ndPHB\video_clip_test_ref.csv')
    video_folder = r'D:\LiDAR_Data\2ndPHB\Video\IntesectionVideo'
    date_column_name = 'DateTime'
    frame_column_name = 'FrameIndex'
    output_name_column = 'SaveName'
    time_interval = 5

    target_frames,video_paths_,output_names_ = analyze_availability(video_folder,ref_table,date_column_name, frame_column_name, output_name_column, time_interval, fps = 10)
    print(target_frames)
    print(video_paths_)
    print(output_names_)