import numpy as np
# from shapely.geometry import LineString,Point, Polygon, shape
# from shapely.ops import unary_union
# import geopandas as gpd
import pandas as pd
import os
import dpkt

from LiDARBase import *
from Utils import *
from GenBckFile import *
# use sobel filter to get horizontal gradient
from p_tqdm import p_umap
from DDBSCAN import *
from functools import partial


# unix time to utc time to pacific time
def unix2utc(unix_time):
    return pd.to_datetime(unix_time,unit='s').tz_localize('UTC').tz_convert('US/Pacific')
def load_pcap(file_path):
    try:
        fpcap = open(file_path, 'rb')
        eth_reader = dpkt.pcap.Reader(fpcap)
    except Exception as ex:
        print(str(ex))
        return None
    return eth_reader
    
def read_packets_offline(pcap_file_path):
    eth_reader = load_pcap(pcap_file_path)
    while True:
        # Simulate reading a packet from the Ethernet
        try:
            ts,buf = next(eth_reader)
        except StopIteration:
            return None
        eth = dpkt.ethernet.Ethernet(buf)
        if eth.type == 2048: # for ipv4
            if (type(eth.data.data) == dpkt.udp.UDP):# for ipv4
                data = eth.data.data.data
                packet_status = eth.data.data.sport
                if packet_status == 2368:
                    if len(data) != 1206:
                        continue
            # raw_packet = np.random.rand(20000,2) * 600  # Placeholder for actual packet data
                    yield (ts,data)
                    
def parse_packets(packet_gen):
    
    culmulative_azimuth_values = []
    culmulative_laser_ids = []
    culmulative_distances = []
    # culmulative_intensities = []
    Td_map = np.zeros((32,1800))
    # Intens_map = np.zeros((32,1800))
    next_ts = 0
    packet = next(packet_gen)
    if packet is None:
        return None
    ts,raw_packet = packet
    distances,intensities,azimuth_per_block,Timestamp = parse_one_packet(raw_packet)
    next_ts = ts + 0.1 # 0.1sec
    azimuth = calc_precise_azimuth(azimuth_per_block) # 32 x 12
    culmulative_azimuth_values.append(azimuth)
    culmulative_laser_ids.append(laser_id)
    culmulative_distances.append(distances)
            
    break_flag = False
    while True:
        if break_flag:
            break  
        while True:
            try:
                packet = next(packet_gen)
            except StopIteration:
                break_flag = True
                break
            ts,raw_packet = packet
            # Placeholder for parsing logic; here we just pass the data through
            distances,intensities,azimuth_per_block,Timestamp = parse_one_packet(raw_packet)
            # flag = self.if_rollover(azimuth_per_block,Initial_azimuth)
            azimuth = calc_precise_azimuth(azimuth_per_block) # 32 x 12
            
            if ts > next_ts:
                
                if len(culmulative_azimuth_values) > 0:
                    
                    culmulative_azimuth_values = np.concatenate(culmulative_azimuth_values,axis = 1)
                    culmulative_azimuth_values += Data_order[:,1].reshape(-1,1)
                    culmulative_laser_ids = np.concatenate(culmulative_laser_ids,axis = 1).flatten()
                    culmulative_distances = np.concatenate(culmulative_distances,axis = 1).flatten()
                    # culmulative_intensities = np.concatenate(culmulative_intensities,axis = 1).flatten()
                    culmulative_azimuth_inds = np.around(culmulative_azimuth_values/0.2).astype('int').flatten()
                    culmulative_azimuth_inds[(culmulative_azimuth_inds<0)|(culmulative_azimuth_inds>1799)] = culmulative_azimuth_inds[(culmulative_azimuth_inds<0)|(culmulative_azimuth_inds>1799)]%1799

                    Td_map[culmulative_laser_ids,culmulative_azimuth_inds] = culmulative_distances
                    # Intens_map[culmulative_laser_ids,culmulative_azimuth_inds] = culmulative_intensities
                    
                    yield Td_map[arg_omega,:] #32*1800
                else:
                    yield Td_map #32*1800

                culmulative_azimuth_values = []
                culmulative_laser_ids = []
                culmulative_distances = []
                # culmulative_intensities = []

                Td_map = np.zeros((32,1800))
                # Intens_map = np.zeros((32,1800))
                next_ts += 0.1
                break
            else:
                culmulative_azimuth_values.append(azimuth)
                culmulative_laser_ids.append(laser_id)
                culmulative_distances.append(distances)
                # culmulative_intensities.append(intensities)
    return None

def get_occupation_ind(Td_map,lane_unit_range_ranging_Tdmap,count_thred,bck_radius, thred_map):
    Foreground_map = ~(np.abs(Td_map - thred_map) <= bck_radius).any(axis = 0)
    Foreground_map = Foreground_map.astype(int)
    Td_map_cos = np.cos(theta * np.pi/180).reshape(-1,1) * Td_map
    occupation_ind = []
    for min_ind,max_ind,min_dis,max_dis in lane_unit_range_ranging_Tdmap:
        if max_ind - min_ind > 900:
            occupation_map = np.concatenate([Foreground_map[:,:min_ind],Foreground_map[:,max_ind:]],axis = 1)
            dis_map = np.concatenate([Td_map_cos[:,:min_ind],Td_map_cos[:,max_ind:]],axis = 1)
        else:
            occupation_map = Foreground_map[:,min_ind:max_ind]
            dis_map = Td_map_cos[:,min_ind:max_ind]
        
        dis_map_ = dis_map * occupation_map
        occupation_flag = ((dis_map_ < max_dis) * (dis_map_ > min_dis)).sum() > count_thred
        occupation_ind.append(occupation_flag)
    occupation_ind = np.array(occupation_ind)
    return occupation_ind

def main(pcap_file_path,lane_drawer,save_path):
    Td_maps = []
    # pcap_file_path = r'D:\LiDAR_Data\9thVir\2024-03-14-23-30-00.pcap'
    # pcap_file_path = r'D:\LiDAR_Data\9thVir\2024-03-14-23-00-00.pcap'
    out_folder_name = os.path.basename(pcap_file_path).split('.')[0]
    packets_gen = read_packets_offline(pcap_file_path)
    packet = next(packets_gen)
    Initial_ts,raw_packet = packet

    packets_gen = read_packets_offline(pcap_file_path)
    td_gen = parse_packets(packets_gen)
    for Td_map in tqdm(td_gen):
        Td_maps.append(Td_map)
    bck_radius = 0.3
    vertical_limits = [0,31]
    lane_drawer = LaneDrawer() # lane drawer for queue detection
    lane_drawer.update_lane_gdf()
    lane_gdf = lane_drawer.lane_gdf
    Td_maps = np.array(Td_maps)

    thred_map = gen_bckmap(Td_maps, N = 10,d_thred = 0.1,bck_n = 3)   
    # np.save('./thred_map.npy',thred_map)
    # thred_map = np.load('./thred_map.npy')
    # create lane_unit_range_ranging_Tdmap
    lane_unit_cos_dis_Tdmap = np.zeros((2,32,1800))
    lane_unit_index_Tdmap = -1 * np.ones((32,1800), dtype = np.int16)
    lane_unit_range_ranging_Tdmap = []
    for unit_ind in range(len(lane_gdf)):
        lane_unit = lane_gdf.iloc[unit_ind]
        x_coords,y_coords = lane_unit.geometry.exterior.coords.xy
        coords = np.c_[x_coords,y_coords]
        
        azimuth_unit = np.arctan2(coords[:,0],coords[:,1]) * 180 / np.pi
        azimuth_unit[azimuth_unit < 0] += 360
        min_azimuth = np.min(azimuth_unit)
        max_azimuth = np.max(azimuth_unit)
        min_ind = int(min_azimuth / 0.2)
        max_ind = int(max_azimuth / 0.2)

        dis = np.sqrt(np.sum(coords**2,axis = 1))

        min_dis = np.min(dis)
        max_dis = np.max(dis)

        if max_ind - min_ind > 900:
            lane_unit_cos_dis_Tdmap[0,:,:min_ind] = min_dis
            lane_unit_cos_dis_Tdmap[0,:,max_ind:] = min_dis
            lane_unit_cos_dis_Tdmap[1,:,:min_ind] = max_dis
            lane_unit_cos_dis_Tdmap[1,:,max_ind:] = max_dis
            lane_unit_index_Tdmap[:,min_ind:] = unit_ind
            lane_unit_index_Tdmap[:,:max_ind] = unit_ind
        else:
            lane_unit_cos_dis_Tdmap[0,:,min_ind:max_ind] = min_dis
            lane_unit_cos_dis_Tdmap[1,:,min_ind:max_ind] = max_dis
            lane_unit_index_Tdmap[:,min_ind:max_ind] = unit_ind
        
        lane_unit_range_ranging_Tdmap.append([min_ind,max_ind,min_dis,max_dis])
    count_thred = 5
    time_space_series = [] # t x lane# x lane_section#
    for Td_map in tqdm(Td_maps):
        activation_profile = []
        occupation_ind = get_occupation_ind(Td_map,lane_unit_range_ranging_Tdmap,count_thred,bck_radius, thred_map)

        for lane_id,g in lane_gdf.groupby('lane_id'):
            activation_profile.append(occupation_ind[g.index])
        time_space_series.append(activation_profile)
    # save_path = r'D:\TimeSpaceDiagramDataset\9th&VirTestDiagram_res5_2300'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for lane_ind in lane_gdf.lane_id.unique():
        lane_activation_profile = []
        for t in range(len(time_space_series)):
            lane_activation_cur = time_space_series[t][lane_ind]
            lane_activation_profile.append(lane_activation_cur)
        lane_activation_profile = np.array(lane_activation_profile,dtype=int)
        lane_activation_profile_T = lane_activation_profile.T
        half_time_folder = os.path.join(save_path,out_folder_name)
        if not os.path.exists(half_time_folder):
            os.makedirs(half_time_folder)
        np.save(os.path.join(half_time_folder,'lane_{}.npy'.format(lane_ind)),lane_activation_profile_T)

if __name__ == '__main__':

    lane_drawer = LaneDrawer() # lane drawer for queue detection
    lane_drawer.update_lane_gdf()

    input_folder = r'D:\LiDAR_Data\9thVir'
    out_folder = r'D:\TimeSpaceDiagramDataset\9th&vir_314'
    pcap_path_list = [os.path.join(input_folder,file) for file in os.listdir(input_folder) if file.endswith('.pcap')]
    p_umap(partial(main,lane_drawer = lane_drawer,save_path = out_folder), pcap_path_list,num_cpus = 5)
