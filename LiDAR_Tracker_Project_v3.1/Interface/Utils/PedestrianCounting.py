import pandas as pd
import numpy as np
import os
import dpkt
from p_tqdm import p_umap
from functools import partial
from threading import Thread
import pickle



rf_classifier = pickle.load(open('./Utils/PedestrianCounter/counter.pkl', 'rb'))


def process_single_traj_file(input_traj_path, save_path, rf_classifier):
    """
    Count pedestrian counts in trajectorie
    Args:
        input_traj_path (str): Path to the trajectory file.
        save_path (str): Path to save the result.
        rf_classifier (RandomForestClassifier): Random forest classifier for counting pedestrians.
    """
    traj = pd.read_csv(input_traj_path)
    ped_index = traj[traj['Class'] == 2].index
    if len(ped_index) != 0:
        ped_num = rf_classifier.predict(np.array(traj.loc[ped_index,['Point_Cnt','Dis','Area']]))
        traj['Ped_Num'] = 0
        traj.loc[ped_index,'Ped_Num'] = ped_num
    
    traj.to_csv(save_path,index = False)

