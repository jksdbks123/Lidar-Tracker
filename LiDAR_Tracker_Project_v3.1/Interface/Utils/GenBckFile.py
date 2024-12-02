import numpy as np
from LiDARBase import *
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

def gen_bckmap(pcap_file_path, N, d_thred, bck_n,termination_event):

    packets_gen = read_packets_offline(pcap_file_path)
    frame_generator = parse_packets(packets_gen)
    aggregated_maps = []
    for Td_map in tqdm(frame_generator):
        aggregated_maps.append(Td_map)
    aggregated_maps = np.array(aggregated_maps)

    if termination_event.is_set():
            return None  # Terminate task
    
    thred_map = np.zeros((bck_n,32,1800))
    for i in range(thred_map.shape[1]):
        for j in range(thred_map.shape[2]):
            thred_map[:,i,j] = get_thred(aggregated_maps[:,i,j],N = N,d_thred = d_thred,bck_n = bck_n)
    
    return thred_map






