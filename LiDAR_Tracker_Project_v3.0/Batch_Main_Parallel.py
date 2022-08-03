import argparse
from ast import arg
from MOT_TD_BCKONLIONE import MOT
from Utils import *
import pandas as pd
import json
import os
from p_tqdm import p_umap
from functools import partial
import time

def run_mot(mot,ref_LLH,ref_xyz,utc_diff):
    mot.initialization()
    if mot.thred_map is not None:
        mot.mot_tracking()
        save_result(mot.Off_tracking_pool,ref_LLH,ref_xyz,mot.traj_path,mot.start_timestamp, utc_diff)
        print(mot.traj_path)

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description='This is a program generating trajectories from .pcap files')
    parser.add_argument('-i','--input', help='path to the folder contains .pcap files and Calibration folder', required=True)
    parser.add_argument('-o','--output', help='specified output path', required=True)
    parser.add_argument('-c','--n_cpu', help='specified CPU number', default = 20 , required=False, type=int)
    parser.add_argument('-d','--utcd', help='Time zone difference to UTC', default = -8 , required=False, type=int)
    parser.add_argument('-t','--timer', help='Timer (hour)', default = 0 , required=False, type=int)
    args = parser.parse_args()
    timer = args.timer
    time.sleep(timer * 3600)

    input_path = args.input
    calibration_path = os.path.join(input_path,'Calibration')
    dir_lis = os.listdir(input_path)
    output_file_path = args.output
    output_traj_path = os.path.join(output_file_path,'Trajectories')
    if not os.path.exists(output_traj_path):
        os.mkdir(output_traj_path)
    traj_list = os.listdir(output_traj_path)
    pcap_paths = []
    pcap_names = []
    for f in dir_lis:
        if 'pcap' in f.split('.'):
            pcap_names.append(f)
            pcap_path = os.path.join(input_path,f)
            pcap_paths.append(pcap_path)

    if len(pcap_paths) == 0:
        print('Pcap file is not detected')
    utc_diff = eval(args.utcd)
    config_path = os.path.join(calibration_path,'config.json')
    ref_LLH_path,ref_xyz_path = os.path.join(calibration_path,'LLE_ref.csv'),os.path.join(calibration_path,'xyz_ref.csv')
    ref_LLH,ref_xyz = np.array(pd.read_csv(ref_LLH_path)),np.array(pd.read_csv(ref_xyz_path))
    if len(np.unique(ref_xyz[:,2])) == 1:
        np.random.seed(1)
        offset = np.random.normal(-0.521,3.28,len(ref_LLH))
        ref_xyz[:,2] += offset
        ref_LLH[:,2] += offset * 3.2808
    ref_LLH[:,[0,1]] = ref_LLH[:,[0,1]] * np.pi/180
    ref_LLH[:,2] = ref_LLH[:,2]/3.2808
    
    with open(config_path) as f:
        params = json.load(f)

        mots = []
        for i,p in enumerate(pcap_paths):
            f_name = pcap_names[i].split('.')[0] + '.csv'
            if f_name in traj_list:
                continue
            out_path = os.path.join(output_traj_path, f_name)
            mots.append(MOT(p,out_path,**params,if_vis=False))
            print(out_path)
            
        n_cpu = args.n_cpu
        print(f'Parallel Processing Begin with {n_cpu} Cpu(s)')
        p_umap(partial(run_mot,ref_LLH = ref_LLH, ref_xyz = ref_xyz, utc_diff = utc_diff), mots,num_cpus = n_cpu)

