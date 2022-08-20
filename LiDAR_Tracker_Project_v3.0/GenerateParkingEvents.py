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

def gen_xyz(dis,i,j):
    longitudes = theta[i]*np.pi / 180
    latitudes = azimuths[j] * np.pi / 180 
    hypotenuses = dis * np.cos(longitudes)
    X = hypotenuses * np.sin(latitudes)
    Y = hypotenuses * np.cos(latitudes)
    Z = dis * np.sin(longitudes)
    return np.array([X,Y,Z])



window_size = 100 # 900 frames -> 9 secs 
sampled_inds = np.arange(0, len(18000), window_size)
Laser_ID = np.arange(57600).reshape((32,1800))

def gen_occ_map(pcap_path,out_path,thred_map,T):
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

    XYZs = []
    dises = []
    Laser_IDs = []
    End = []
    Start = [] 
    for laser_id in range(aggregated_map.shape[1]):
        for azimuth_channel in range(aggregated_map.shape[2]):

            temp = aggregated_map[:,laser_id,azimuth_channel].copy()
            bck_dis = thred_map[:,laser_id,azimuth_channel].max()

            Parking_ind = []
            Sampled_ind = []
            for i in range(1,len(sampled_inds)):
                cur_dis = temp[sampled_inds[i]]
                if cur_dis == 0:
                    Parking_ind.append(False)
                else:
                    if (bck_dis - cur_dis) > 1.5:
                        past_dis = temp[sampled_inds[i] - window_size:sampled_inds[i]]
                        past_dis = past_dis[past_dis != 0]
                        if len(past_dis) == 0:
                            Parking_ind.append(False)
                        else:
                            if np.abs(cur_dis - np.median(past_dis)) < 1:
                                Parking_ind.append(True)
                            else:
                                Parking_ind.append(False)
                    else:
                        Parking_ind.append(False)
                Sampled_ind.append(i)

            Parking_ind = np.array([Parking_ind[0]] + Parking_ind)
            counts,appears = count(~Parking_ind)
            if len(counts) > 0:
                for i,a in enumerate(appears):
                    if counts[i] < 6: # if the parking is less than 60 sec will not be recorded
                        Parking_ind[a: a + counts[i]] = True
                counts,appears = count(Parking_ind)

                for i,a in enumerate(appears):
                    c = counts[i]
                    dis = temp[sampled_inds[a]]
                    Laser_IDs.append(Laser_ID[laser_id,azimuth_channel])
                    XYZ = gen_xyz(dis,laser_id,azimuth_channel)
                    XYZs.append(XYZ)
                    Start.append(sampled_inds[a])
                    End.append(sampled_inds[a + c - 1])
    XYZs = np.array(XYZs)
    Laser_IDs = np.array(Laser_IDs).reshape(-1,1)
    Start = np.array(Start).reshape(-1,1)
    End = np.array(End).reshape(-1,1)
    LLH = convert_LLH(XYZs.astype(np.float64),T)
    ts_arr = f_name.split('.')[0].split('-')
    f_name = '2022-1-22-12-0-0.pcap'
    ts_arr = f_name.split('.')[0].split('-')
    Day = pd.Timestamp(eval(ts_arr[0]),eval(ts_arr[1]),eval(ts_arr[2]),0,0,0)
    try:
        sec = eval(ts_arr[5])
    except:
        sec = eval(ts_arr[5][:-1])
    f_time = pd.Timestamp(eval(ts_arr[0]),eval(ts_arr[1]),eval(ts_arr[2]),eval(ts_arr[3]),eval(ts_arr[4]),sec)
    Start_ts = f_time + pd.Series([pd.Timedelta(seconds = Start[i][0]/10) for i in range(len(Start))])
    End_ts = f_time + pd.Series([pd.Timedelta(seconds = End[i][0]/10) for i in range(len(End))])
    
    Result_info = np.concatenate([Laser_IDs,LLH,XYZs,Start,End],axis = 1)
    Result_info = pd.DataFrame(Result_info, columns = ['LaserBeamID','Lon','Lat','Elev','X','Y','Z','Start_frame','End_frame'])
    Result_info.insert(column= 'End_ts', value = End_ts, loc = 0)
    Result_info.insert(column= 'Start_ts', value = Start_ts, loc = 0)

    out_path = out_path + '.csv'
    Result_info.to_csv(out_path,index = False)


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
