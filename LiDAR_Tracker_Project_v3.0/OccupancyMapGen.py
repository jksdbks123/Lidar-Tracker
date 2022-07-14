import argparse
from ast import arg
from pyrsistent import T
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

def gen_occ_map(pcap_path,T,out_path):
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
    thred_map = gen_bckmap(aggregated_map, N = 10,d_thred = 0.08,bck_n = 3 )
    aggregated_Labeling_map = []
    for i in range(aggregated_map.shape[0]):
        Td_map = aggregated_map[i]
        Foreground_map = ~(np.abs(Td_map - thred_map) <= 1.5).any(axis = 0)
        Labeling_map = db.fit_predict(Td_map= Td_map,Foreground_map=Foreground_map)
        aggregated_Labeling_map.append(Labeling_map)
    aggregated_Labeling_map = np.array(aggregated_Labeling_map)
    points = []
    occupancies = []
    row_ind = []
    col_ind = []
    for i in range(32):
        for j in range(1800):
            foreground_ind = aggregated_Labeling_map[:,i,j] != -1
            if foreground_ind.any(): # foreground
                
                dis_values = aggregated_map[foreground_ind,i,j]
                labels = dbscan.fit_predict(dis_values.reshape(-1,1))
                unique_labels = np.unique(labels)
                unique_labels = unique_labels[1:]
                if len(unique_labels) >= 1: 
                    for l in unique_labels:
                        dis = np.mean(dis_values[labels == l])
                        XYZ = gen_xyz(dis,i,j)
                        points.append(XYZ)
                        occupancy = (labels == l).sum()/len(foreground_ind)
                        occupancies.append(occupancy)
                        row_ind.append(i)
                        col_ind.append(j)
    points = np.array(points)
    occupancies = np.array(occupancies)
    col_ind = np.array(col_ind)
    row_ind = np.array(row_ind)
    pcd = get_pcd_uncolored(aggregated_map[2])
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.3,
                                            ransac_n=10,
                                            num_iterations=1000)
    height = np.abs(plane_model[0] * points[:,0] + plane_model[1] * points[:,1] + plane_model[2] * points[:,2] + plane_model[3]) / (np.sqrt(plane_model[0]**2 + plane_model[1]**2 +plane_model[2]**2))
    LLH = convert_LLH(points.astype(np.float64),T)
    resultGram = pd.DataFrame(np.concatenate([points,LLH,occupancies.reshape(-1,1),height.reshape(-1,1),row_ind.reshape(-1,1),col_ind.reshape(-1,1)],axis =1 ),columns=['X','Y','Z','Longitude','Latitude','Elevation','Occupancy','Height','LaserID','AzimuthID'])
    resultGram.to_csv(out_path,index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a program generating occupancy map from .pcap files')
    parser.add_argument('-i','--input', help='path to the folder contains .pcap files', required=True)
    parser.add_argument('-o','--output', help='specified output path', required=True)
    parser.add_argument('-c','--n_cpu', help='specified CPU number', default = 20 , required=False, type=int)
    args = parser.parse_args()

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

    out_paths = []
    for i,p in enumerate(pcap_paths):
        f_name = pcap_names[i].split('.')[0] + '.csv'
        if f_name in traj_list:
            continue
        out_path = os.path.join(output_traj_path, f_name)
        out_paths.append(out_path)
        print(out_path)

    n_cpu = args.n_cpu
    print(f'Parallel Processing Begin with {n_cpu} Cpu(s)')
    p_umap(partial(gen_occ_map,T = T), pcap_paths,out_paths,num_cpus = n_cpu)
