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
from tqdm import tqdm

from Utils import *
from tqdm import tqdm

def get_thred(temp,N = 10,d_thred = 0.1,bck_n = 3):
    temp = temp.copy()
    total_sample = len(temp)
    bck_ds = []
    bck_portions = []
    repeat = 0
    while repeat < N:
        if len(temp) == 0:
            break
        sample = np.random.choice(temp,replace=False)
        ind = np.abs(temp - sample) < 0.3
        portion = ind.sum()/total_sample
        if portion > d_thred:
            bck_portions.append(portion)
            bck_ds.append(sample)
            temp = temp[~ind]
        repeat += 1
        
    bck_ds = np.array(bck_ds)
    bck_portions = np.array(bck_portions)
    arg_ind = np.argsort(bck_portions)[::-1]
    bck_ds_ = bck_ds[arg_ind[:bck_n]]
    
    if len(bck_ds_) <= bck_n:
        bck_ds_ = np.concatenate([bck_ds_,-1 * np.ones(bck_n - len(bck_ds_))])
    return bck_ds_

def gen_bckmap(aggregated_maps, N, d_thred, bck_n):
    thred_map = np.zeros((bck_n,32,1800))
    for i in range(thred_map.shape[1]):
        for j in range(thred_map.shape[2]):
            thred_map[:,i,j] = get_thred(aggregated_maps[:,i,j],N = N,d_thred = d_thred,bck_n = bck_n)

    return thred_map

def generate_and_save_background(background_data):
    thred_map = gen_bckmap(np.array(background_data), N = 10,d_thred = 0.1,bck_n = 3)
    np.save('./config_files/thred_map.npy',thred_map)
    print('Generate Bck')



def gen_agg_map(pcap_path,out_path):
    aggregated_map = []
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
    if len(aggregated_map) >= 2000:
        selected_length = 2000
    else:
        selected_length = len(aggregated_map)
    if selected_length != 0:
        selected_inds = np.random.choice(np.arange(len(aggregated_map)),selected_length,replace = False)
        selected_maps = aggregated_map[selected_inds]
        np.save(out_path,selected_maps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a program generating background map from .pcap files')
    parser.add_argument('-i','--input', help='path to the folder contains .pcap files', required=True)
    parser.add_argument('-o','--output', help='specified output path', required=True)
    parser.add_argument('-c','--n_cpu', help='specified CPU number', default = 20 , required=False, type=int)
    args = parser.parse_args()

    input_path = args.input
    dir_lis = os.listdir(input_path)
    output_file_path = args.output
    output_traj_path = os.path.join(output_file_path,'Result')
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

    print('Generating Background')
    f_path = output_traj_path
    agg_dirs = np.array(os.listdir(f_path))
    hours = [eval(f.split('-')[3]) for f in agg_dirs]
    agg_dirs = agg_dirs[np.argsort(hours)]
    aggregated_maps = []
    for f in tqdm(agg_dirs):
        aggregated_map = np.load(os.path.join(f_path,f))
        aggregated_maps.append(aggregated_map)
    aggregated_maps = np.concatenate(aggregated_maps)

    output_file_path = args.output
    output_traj_path = os.path.join(output_file_path,'Thred_map')
    if not os.path.exists(output_traj_path):
        os.mkdir(output_traj_path)
    
    thred_map = gen_bckmap(aggregated_maps, N = 20,d_thred = 0.12,bck_n = 5)
    np.save(os.path.join(output_traj_path,'bck_map.npy'),thred_map)