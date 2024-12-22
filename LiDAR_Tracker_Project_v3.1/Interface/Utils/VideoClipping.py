import cv2
from p_tqdm import p_umap
from functools import partial
from threading import Thread
import os

def clip_single_video(input_path, output_path, start_frame, end_frame):
    """
    Clips a video using OpenCV based on start and end timestamps.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the clipped video.
        start_time (float): Start time in seconds.
        end_time (float): End time in seconds.
    """
    # Open the video file
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the frame
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Height of the frame
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec

    # Set up VideoWriter for saving the output
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Set the starting position of the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Loop through frames and save to the output
    current_frame = start_frame
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached or error reading frame.")
            break

        out.write(frame)
        current_frame += 1

    # Release resources
    cap.release()
    out.release()

def batch_video_clipping(input_folder, output_folder, start_frames, end_frames):
    """
    Batch clips videos in a folder based on start and end frames.

    Args:
        input_folder (str): Path to the folder containing input videos.
        output_folder (str): Path to save the clipped videos.
        start_frames (list): List of start frames for each video.
        end_frames (list): List of end frames for each video.
    """
    # Get a list of video files in the input folder
    video_files = [f for f in os.listdir(input_folder) if f.endswith('.avi')]

    if not video_files:
        print("No video files found in the input folder.")
        return

    p_umap(
        partial(clip_single_video, output_path=output_folder),
        video_files,
        [os.path.join(input_folder, f) for f in video_files],
        start_frames,
        end_frames
    )
def run_batch_video_clipping_threaded(input_folder, output_folder, start_frames, end_frames):
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
        args=(input_folder, output_folder, start_frames, end_frames)
    )
    thread.start()
    

if __name__ == "__main__":
    pass