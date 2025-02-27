
import os
import dpkt
from p_tqdm import p_umap
from functools import partial
from threading import Thread
import pickle






def process_single_traj_file(input_traj_path, output_traj_path, point_number_name_column, area_name_column, distance_name_column, rf_classifier):
    """
    Count pedestrian counts in trajectorie
    Args:
        input_traj_path (str): Path to the trajectory file.
        save_path (str): Path to save the result.
        rf_classifier (RandomForestClassifier): Random forest classifier for counting pedestrians.
    """
    import pandas as pd
    import numpy as np
    traj = pd.read_csv(input_traj_path)
    ped_index = traj[traj['Class'] == 2].index
    if len(ped_index) != 0:
        ped_num = rf_classifier.predict(np.array(traj.loc[ped_index,[point_number_name_column,distance_name_column,area_name_column]]))
        traj['Ped_Num'] = 0
        traj.loc[ped_index,'Ped_Num'] = ped_num
    
    traj.to_csv(output_traj_path,index = False)

def batch_traj_processing(output_folder,traj_folder, point_number_name_column, area_name_column, distance_name_column, n_cpu):
    """
    Batch counting pedestians in trajectories.
    Args:
        traj_folder (str): Path to the folder containing trajectory files.
        output_folder (str): Path to save the clipped videos.
        start_frames (list): List of start frames for each video.
        end_frames (list): List of end frames for each video.
    """
    rf_classifier = pickle.load(open('./Utils/PedestrianCounter/counter.pkl', 'rb'))
    traj_path_list = [os.path.join(traj_folder, f) for f in os.listdir(traj_folder) if f.endswith('.csv')]
    output_traj_path_list = [os.path.join(output_folder, f) for f in os.listdir(traj_folder) if f.endswith('.csv')]
    p_umap(
        partial(process_single_traj_file, point_number_name_column = point_number_name_column, area_name_column = area_name_column, distance_name_column = distance_name_column, rf_classifier = rf_classifier),
        traj_path_list,
        output_traj_path_list,
        num_cpus = n_cpu
    )
def run_batch_traj_processing_threaded(output_folder,traj_folder, point_number_name_column, area_name_column, distance_name_column, n_cpu):
    """
    Batch counting pedestians in trajectories.
    Args:
        traj_folder (str): Path to the folder containing trajectory files.
        output_folder (str): Path to save the clipped videos.
        start_frames (list): List of start frames for each video.
        end_frames (list): List of end frames for each video.
    """
    thread = Thread(
        target=batch_traj_processing,
        args=(output_folder,traj_folder, point_number_name_column, area_name_column, distance_name_column, n_cpu),
    )
    thread.start()

if __name__ == "__main__":
    out_folder = r'D:\LiDAR_Data\Harvard_out_ped_count'
    traj_folder = r'D:\LiDAR_Data\Harvard_out'
    point_number_name_column = 'Point_Cnt'
    area_name_column = 'Area'
    distance_name_column = 'Dis'
    n_cpu = 4
    run_batch_traj_processing_threaded(out_folder, traj_folder, point_number_name_column, area_name_column, distance_name_column, n_cpu)