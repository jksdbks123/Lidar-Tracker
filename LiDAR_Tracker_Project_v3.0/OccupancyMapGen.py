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

def count(TSAv):
    temp_count = 0
    apear_ind = []
    counts = []
    for i in range(len(TSAv)):
        if (TSAv[i] == True):
            temp_count += 1
        else:
            if (i > 0) & (TSAv[i - 1] == True):
                apear_ind.append(i - temp_count)
                counts.append(temp_count)
                temp_count = 0
                counts.append(0)
            else:
                counts.append(0)
        if (i == len(TSAv) - 1) & (temp_count != 0):
            apear_ind.append(i - temp_count + 1)
            counts.append(temp_count)
    counts = np.array(counts)
    counts = counts[counts > 0]
    return np.array(counts), np.array(apear_ind)

def get_parking(temp,N = 20,d_thred = 0.15,bck_n = 6):
    temp = temp.copy()
    total_sample = len(temp)
    temp = temp[temp > 0]
    bck_ds = []
    bck_portions = []
    repeat = 0
    while repeat < N:
        if len(temp) == 0:
            break
        sample = np.random.choice(temp,replace=False)
        
        ind = np.abs(temp - sample) < 0.4
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

def gen_xyz(dis,i,j):
    longitudes = theta[i]*np.pi / 180
    latitudes = azimuths[j] * np.pi / 180 
    hypotenuses = dis * np.cos(longitudes)
    X = hypotenuses * np.sin(latitudes)
    Y = hypotenuses * np.cos(latitudes)
    Z = dis * np.sin(longitudes)
    return np.array([X,Y,Z])

db = Raster_DBSCAN(window_size=[5,13],eps = 1.5,min_samples = 12,Td_map_szie = [32,1800])
dbscan = DBSCAN(eps = 1, min_samples = 20)

def gen_occ_map(pcap_path,out_path,thred_map,T):
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
    Parking_coord = np.array([[15,496],[11,537],[14,564]])
    for pc_i,pc in enumerate(Parking_coord):
        row_ind,col_ind = [],[]
        occupancies = []
        starts = []
        ends = []
        points = []
        laser_id = pc[0]
        azimuth_id = pc[1]

        temp = aggregated_map[:,laser_id,azimuth_id].copy()
        thred = thred_map[:,laser_id,azimuth_id]
        thred_max = thred.max()
        bck_inds = ((np.abs((temp - thred_max)) < 2))
        temp[bck_inds] = 0
        time_window = 600
        parking_label  = []
        for i in range(time_window,len(temp)):
            past_dis = temp[i - time_window:i]
            past_dis = past_dis[past_dis!=0]
            if len(past_dis) == 0:
                parking_label.append(False)
            else:
                if np.abs(temp[i] - np.median(past_dis)) < 0.6:
                    parking_label.append(True)
                else:
                    parking_label.append(False)
        parking_label = time_window*[parking_label[0]] + parking_label
        parking_label = np.array(parking_label)
        counts,appears = count(~parking_label)
        for i,a in enumerate(appears):
            c = counts[i]
            if c < time_window:
                parking_label[a:a+c+1] = True
        counts,appears = count(parking_label)
        for i,a in enumerate(appears):
            c = counts[i]
            if c < time_window:
                parking_label[a:a+c+1] = False
        counts,appears = count(parking_label) 

        for l,a in enumerate(appears):
            parking_dis = temp[a:a+counts[l]]
            parking_dis = parking_dis[parking_dis!=0]
            dis =  np.median(parking_dis)
            XYZ = gen_xyz(dis,i,j)
            points.append(XYZ)
            occupancy = counts[l]
            occupancies.append(occupancy)
            row_ind.append(i)
            col_ind.append(j)
            starts.append(a)
            ends.append(a + counts[l])

        points = np.array(points)
        occupancies = np.array(occupancies)
        col_ind = np.array(col_ind)
        row_ind = np.array(row_ind)
        starts = np.array(starts)
        ends = np.array(ends)
        
        LLH = convert_LLH(points.astype(np.float64),T)
        resultGram = pd.DataFrame(np.concatenate([points,LLH,occupancies.reshape(-1,1),starts.reshape(-1,1),ends.reshape(-1,1),row_ind.reshape(-1,1),col_ind.reshape(-1,1)],axis =1 ),columns=['X','Y','Z','Longitude','Latitude','Elevation','Occupancy','Starts','Ends','LaserID','AzimuthID'])
        out_path = out_path + '_{}'.format(pc_i) + '.csv'
        resultGram.to_csv(out_path,index = False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a program generating occupancy map from .pcap files')
    parser.add_argument('-i','--input', help='path to the folder contains .pcap files', required=True)
    parser.add_argument('-o','--output', help='specified output path', required=True)
    parser.add_argument('-c','--n_cpu', help='specified CPU number', default = 20 , required=False, type=int)
    args = parser.parse_args()

    input_path = args.input
    calibration_path = os.path.join(input_path,'Calibration')
    thred_map = np.load(os.path.join(calibration_path,'bck_larue.npy'))

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

    # config_path = os.path.join(calibration_path,'config.json')
    ref_LLH_path,ref_xyz_path = os.path.join(calibration_path,'LLE_ref.csv'),os.path.join(calibration_path,'xyz_ref.csv')
    ref_LLH,ref_xyz = np.array(pd.read_csv(ref_LLH_path)),np.array(pd.read_csv(ref_xyz_path))
    if len(np.unique(ref_xyz[:,2])) == 1:
        np.random.seed(1)
        offset = np.random.normal(-0.521,3.28,len(ref_LLH))
        ref_xyz[:,2] += offset
        ref_LLH[:,2] += offset * 3.2808
    ref_LLH[:,[0,1]] = ref_LLH[:,[0,1]] * np.pi/180
    ref_LLH[:,2] = ref_LLH[:,2]/3.2808
    T = generate_T(ref_LLH,ref_xyz)
    out_paths = []
    for i,p in enumerate(pcap_paths):
        f_name = pcap_names[i].split('.')[0] 
        if f_name in traj_list:
            continue
        out_path = os.path.join(output_traj_path, f_name)
        out_paths.append(out_path)
        print(out_path)

    n_cpu = args.n_cpu
    print(f'Parallel Processing Begin with {n_cpu} Cpu(s)')
    p_umap(partial(gen_occ_map,thred_map = thred_map,T = T), pcap_paths,out_paths,num_cpus = n_cpu)
