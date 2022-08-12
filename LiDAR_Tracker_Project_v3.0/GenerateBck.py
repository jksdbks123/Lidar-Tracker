import argparse
from ast import arg
from Utils import *
import pandas as pd
import os
from tqdm import tqdm
from functools import partial
from BfTableGenerator import TDmapLoader
from DDBSCAN import Raster_DBSCAN
from sklearn.cluster import DBSCAN
from VisulizerTools import *
from Utils import *

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
    for i in tqdm(range(thred_map.shape[1])):
        for j in range(thred_map.shape[2]):
            thred_map[:,i,j] = get_thred(aggregated_maps[:,i,j],N = N,d_thred = d_thred,bck_n = bck_n)
    return thred_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a program generating background scheme from .npy files')
    parser.add_argument('-i','--input', help='path to the folder contains .pcap files', required=True)
    parser.add_argument('-o','--output', help='specified output path', required=True)
    args = parser.parse_args()

    f_path = parser.input
    agg_dirs = np.array(os.listdir(f_path))
    hours = [eval(f.split('-')[3]) for f in agg_dirs]
    agg_dirs = agg_dirs[np.argsort(hours)]
    aggregated_maps = []
    for f in tqdm(agg_dirs):
        aggregated_map = np.load(os.path.join(f_path,f))
        aggregated_maps.append(aggregated_map)
    aggregated_maps = np.concatenate(aggregated_maps)

    output_file_path = args.output
    output_traj_path = os.path.join(output_file_path,'Result')
    if not os.path.exists(output_traj_path):
        os.mkdir(output_traj_path)
    
    thred_map = gen_bckmap(aggregated_maps, N = 20,d_thred = 0.12,bck_n = 5)
    np.save(os.path.join(output_file_path,'bck_larue.npy'),thred_map)
    
    