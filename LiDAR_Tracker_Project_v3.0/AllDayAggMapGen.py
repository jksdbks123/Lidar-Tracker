import argparse
from ast import arg
from Utils import *
import pandas as pd
import os
from p_tqdm import p_umap
from functools import partial
from BfTableGenerator import TDmapLoader
from DDBSCAN import Raster_DBSCAN
from sklearn.cluster import DBSCAN
from VisulizerTools import *
from Utils import *



def gen_agg_map(pcap_path,out_path):
    aggregated_map = []
    # pcap_path = r'D:\LiDAR_Data\ASWS\MtRose\Thomas_asws2nd\2022-04-07-04-30-31.pcap'
    end_frame = 18000
    lidar_reader = TDmapLoader(pcap_path)
    frame_gen = lidar_reader.frame_gen()
    for i in range(end_frame):
        Frame = next(frame_gen)
        if Frame is None:
            break 
        Td_map,Int_map = Frame
        aggregated_map.append(Td_map)
    aggregated_map = np.array(aggregated_map)
    selected_inds = np.random.choice(np.arange(len(aggregated_map)),2000)
    selected_maps = aggregated_map[selected_inds]
    np.save(out_path,selected_maps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a program generating occupancy map from .pcap files')
    parser.add_argument('-i','--input', help='path to the folder contains .pcap files', required=True)
    parser.add_argument('-o','--output', help='specified output path', required=True)
    parser.add_argument('-c','--n_cpu', help='specified CPU number', default = 20 , required=False, type=int)
    args = parser.parse_args()

    input_path = args.input
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

    out_paths = []
    for i,p in enumerate(pcap_paths):
        f_name = pcap_names[i].split('.')[0] + '.npy'
        if f_name in traj_list:
            continue
        out_path = os.path.join(output_traj_path, f_name)
        out_paths.append(out_path)
        print(out_path)

    n_cpu = args.n_cpu
    print(f'Parallel Processing Begin with {n_cpu} Cpu(s)')
    p_umap(partial(gen_agg_map), pcap_paths,out_paths,num_cpus = n_cpu)
