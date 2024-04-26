import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import os
from tqdm import tqdm
# from Utils import get_foreground_point_cloud
from LiDARBase import parse_one_packet,calc_precise_azimuth,laser_id,Data_order,arg_omega,calc_timing_offsets,get_foreground_point_cloud
from GenBckFile import *
# use sobel filter to get horizontal gradient
from scipy import ndimage
from DDBSCAN import *
import statistics
from DataExtractionTools import *
from p_tqdm import p_umap
from functools import partial
from Utils import LaneDrawer



def main(pcap_file_path,lane_drawer,out_path):
    count_thred = 3 # activation point cloud # threshold
    bck_radius = 0.2
    vertical_limits = [0,31]
    # extract the pcap_file name of the pcap_file_path and exclude the .pcap extension
    pcap_file_name = os.path.basename(pcap_file_path).split('.')[0]
    # create a folder to store the output by the pcap_file name and out_path
    out_folder = os.path.join(out_path,pcap_file_name)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    # create a folder in out_folder to store the figures
    fig_folder = os.path.join(out_folder,'figures')
    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)
    print(out_folder,fig_folder)
    Td_maps = []
    packets_gen = read_packets_offline(pcap_file_path)
    packet = next(packets_gen)
    Initial_ts,raw_packet = packet
    # unix time to utc time to pacific time
    Initial_date = unix2utc(Initial_ts)
    packets_gen = read_packets_offline(pcap_file_path)
    td_gen = parse_packets(packets_gen)
    print('Loading Tdmaps')
    for Td_map in tqdm(td_gen):
        if type(Td_map) != np.ndarray:
            continue
        if Td_map.shape != (32,1800):
            continue
        Td_maps.append(Td_map)

    
    background_data = np.array(Td_maps)
    print('Generating Background')
    thred_map = gen_bckmap(np.array(background_data), N = 10,d_thred = 0.1,bck_n = 3)
    # save thred_map as .npy file in out_folder
    np.save(os.path.join(out_folder,'thred_map.npy'),thred_map)
    

    time_space_series = [] # t x laen# x lane_section#
    
    for Td_map in tqdm(Td_maps):
        
        data_raw,point_labels,tracking_dic = get_foreground_point_cloud(thred_map,bck_radius,
                                                                                Td_map,vertical_limits)
        lane_section_foreground_point_counts = get_lane_section_foreground_point_counts(lane_drawer.lane_subsections_poly,
                                                                                    lane_drawer.lane_gdf,
                                                                                    data_raw,point_labels)
        activation_profile = []
        for lane_counts in lane_section_foreground_point_counts:
            occupation = np.array(lane_counts) > count_thred
            activation_profile.append(occupation)
        time_space_series.append(activation_profile)
        
    print('Generating Queue Info')
    for lane_ind in range(len(lane_drawer.lane_subsections_poly)):
        lane_activation_profile = []
        for t in range(len(time_space_series)):
            lane_activation_cur = time_space_series[t][lane_ind]
            lane_activation_profile.append(lane_activation_cur)
        lane_activation_profile = np.array(lane_activation_profile,dtype=int)
        lane_activation_profile_T = lane_activation_profile.T
        stop_specturm = ndimage.sobel(lane_activation_profile_T, 0)  # horizontal gradient
        empty_specturm = np.zeros(lane_activation_profile_T.shape,dtype=np.uint8)
        lines = cv2.HoughLinesP(image= (stop_specturm > 0).astype(np.uint8),rho=1,theta=np.pi/2,threshold=100,minLineLength=50,maxLineGap=5)
        if lines is not None:

            for line in lines:
                x1,y1,x2,y2 = line[0]
                cv2.line(empty_specturm,(x1,y1),(x2,y2),(1,0,0),2)
                        
        queue_length_indicator = empty_specturm != 0
        total_queue_length_curve = []
        for i in range(0,queue_length_indicator.shape[1],1):
            stops = np.where(queue_length_indicator[:,i])[0]
            if len(stops) == 0:
                total_queue_length_curve.append(0)
                continue
            queue_length_t = (queue_length_indicator.shape[0] - stops.min()) * 0.5
            total_queue_length_curve.append(queue_length_t)
        total_queue_length_curve = np.array(total_queue_length_curve)
        # sample at 10 frames
        total_queue_length_curve_sec = total_queue_length_curve[::10]

        # define a kernal to identify a sudden change toward upper direction
        kernel = np.array([0] * 10 + [1] * 5)
        volumes = []
        for i in range(1,10):
            activation = lane_activation_profile_T[-i,:]
            # convolve the kernal with the lane_activation_profile_T
            convolution = np.convolve(activation,kernel,mode='same')
            passing_incator = convolution > 0
            activation_start,activation_end = find_concecutive_activation(passing_incator)
            # activation_end = np.array(activation_end)
            activation_start = np.array(activation_start)
            volume_section = []
            for ts in range(10,lane_activation_profile_T.shape[1]+ 10,10):
                volume_section.append(((activation_start < ts)&(activation_start > ts - 10)).sum())
            volumes.append(volume_section)
        volumes = np.array(volumes)
        volumes_sec = []
        for i in range(volumes.shape[1]):
            volumes_sec.append(statistics.mode(volumes[:,i]))
        volumes_sec = np.array(volumes_sec)
        # save queue curve and volumes_sec as one dataframe and save to out_folder, and named by queue_volume_{lane_ind}.csv
        # use init_date to create the time index at second level
        time_index = pd.date_range(Initial_date, periods = len(total_queue_length_curve_sec), freq = '1s')
        queue_volume_df = pd.DataFrame({'queue_length':total_queue_length_curve_sec,'volume':volumes_sec},index = time_index)
        queue_volume_df.to_csv(os.path.join(out_folder,f'queue_volume_{lane_ind}.csv'))
        
        # times new roman
        plt.rcParams['font.family'] = 'Times new roman'
        cum_volumes = []
        for i in range(1,volumes_sec.shape[0]):
            cum_volumes.append(volumes_sec[:i].sum())
        cum_volumes = np.array(cum_volumes)
        plt.figure(figsize = (10,5))
        plt.grid()
        plt.plot(cum_volumes, linewidth = 2, color = 'black')
        plt.xlabel('Time (1s)',fontsize = 15)
        plt.ylabel('Volume (veh)',fontsize = 15)
        plt.savefig(os.path.join(fig_folder,f'volume_{lane_ind}.png'),dpi = 300)
        plt.close()

        plt.figure(figsize = (10,5))
        plt.plot(total_queue_length_curve_sec, color = 'black')
        plt.xlabel('Time (1s)',fontsize = 15)
        plt.ylabel('Queue Length (m)',fontsize = 15)
        plt.savefig(os.path.join(fig_folder,f'queue_{lane_ind}.png'),dpi = 300)
        plt.close()
        plt.figure(figsize = (10,10))
        # increase distincness of the image
        plt.imshow(empty_specturm,aspect = 'auto',cmap='gray',interpolation = None)
        # plt.imshow(sobel_v,aspect = 'auto',cmap='gray',interpolation = None)
        plt.xlabel('Time (0.1s)',fontsize = 15)
        plt.ylabel('Lane Unit (0.5m)',fontsize = 15)
        # plt.xlim(14000,15500)# plt.ylim(0,32)
        plt.savefig(os.path.join(fig_folder,f'queue_spectrum_{lane_ind}.png'),dpi = 300)
        plt.close()
        plt.figure(figsize = (10,10))
        plt.imshow(lane_activation_profile_T,aspect = 'auto',interpolation = None)
        plt.xlabel('Time (0.1s)',fontsize = 15)
        plt.ylabel('Lane Unit (0.5m)',fontsize = 15)
        # plt.xlim(12000,15500)
        plt.savefig(os.path.join(fig_folder,f'time_space_{lane_ind}.png'),dpi = 300)
        plt.close()
        

if __name__ == '__main__':
    lane_drawer = LaneDrawer() # lane drawer for queue detection
    lane_drawer.update_lane_gdf()
    # pcap_file_path = r'D:\LiDAR_Data\9thVir\2024-03-14-23-30-00.pcap'
    # out_path = 'D:\LiDAR_Data\9thVir_out'
    # main(pcap_file_path,lane_drawer,out_path)


    
    

    input_folder = r'D:\LiDAR_Data\9thVir'
    out_folder = r'D:\LiDAR_Data\9thVir_out'
    pcap_path_list = [os.path.join(input_folder,file) for file in os.listdir(input_folder) if file.endswith('.pcap')]
    p_umap(partial(main,lane_drawer = lane_drawer,out_path = out_folder), pcap_path_list,num_cpus = 4)

    