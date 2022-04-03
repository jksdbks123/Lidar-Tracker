import argparse
from MOT_TD_BCKONLIONE import MOT
from Utils import *
import pandas as pd
import json
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from p_tqdm import p_map,p_umap
from functools import partial

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='This is a program generating trajectories from .pcap files')
    parser.add_argument('-i','--input', help='path that contains .pcap file', required=True)
    parser.add_argument('-o','--output', help='specified output path', required=True)
    args = parser.parse_args()

    input_path = args.input
    calibration_path = os.path.join(input_path,'Calibration')
    dir_lis = os.listdir(input_path)
    output_file_path = args.output
    output_traj_path = os.path.join(output_file_path,'Trajectories')
    if not os.path.exists(output_traj_path):
        os.mkdir(output_traj_path)
    pcap_paths = []

    for f in dir_lis:
        if 'pcap' in f.split('.'):
            pcap_path = os.path.join(input_path,f)
            pcap_paths.append(pcap_path)

    if len(pcap_paths) == 0:
        print('Pcap file is not detected')

    config_path = os.path.join(calibration_path,'config.json')
    ref_LLH_path,ref_xyz_path = os.path.join(calibration_path,'LLE_ref.csv'),os.path.join(calibration_path,'xyz_ref.csv')
    ref_LLH,ref_xyz = np.array(pd.read_csv(ref_LLH_path)),np.array(pd.read_csv(ref_xyz_path))
    if len(np.unique(ref_xyz[:,2])) == 1:
        np.random.seed(1)
        offset = np.random.normal(-0.521,3.28,len(ref_LLH))
        ref_xyz[:,2] += offset
        ref_LLH[:,2] += offset
    ref_LLH[:,[0,1]] = ref_LLH[:,[0,1]] * np.pi/180
    ref_LLH[:,2] = ref_LLH[:,2]/3.2808
    # bck_map_path = os.path.join(calibration_path,'bck_map.npy')
    # bck_map = np.load(bck_map_path)
    # plane_model_path = os.path.join(calibration_path,'plane_model.npy')
    # plane_model = np.load(plane_model_path)
    with open(config_path) as f:
        params = json.load(f)

    
    def run_mot(mot,ref_LLH,ref_xyz):
        mot.initialization()
        mot.mot_tracking()
        save_result(mot.Off_tracking_pool,ref_LLH,ref_xyz,mot.traj_path )
        print(mot.traj_path)
        
    mots = []
    for i,p in enumerate(pcap_paths):
        f_name = p.split('.')[0].split('\\')[-1] +'.csv'
        out_path = os.path.join(output_traj_path, f_name)
        mots.append(MOT(p,out_path,**params))
        print(out_path)

    print('Parallel Processing Begin')

    p_umap(partial(run_mot,ref_LLH = ref_LLH, ref_xyz = ref_xyz), mots,num_cpus = 8)

