from multiprocessing.spawn import import_main_path
from tqdm import tqdm
import matplotlib.pyplot as plt
import dpkt
import numpy as np
import open3d as op3 
import pandas as pd
import os
import time 
import sys 
from VisulizerTools import *
from DDBSCAN import Raster_DBSCAN
class TDmapLoader():
    def __init__(self,file_path): 
        self.Data_order = np.array([[-25,1.4],[-1,-4.2],[-1.667,1.4],[-15.639,-1.4],
                                    [-11.31,1.4],[0,-1.4],[-0.667,4.2],[-8.843,-1.4],
                                    [-7.254,1.4],[0.333,-4.2],[-0.333,1.4],[-6.148,-1.4],
                                    [-5.333,4.2],[1.333,-1.4],[0.667,4.2],[-4,-1.4],
                                    [-4.667,1.4],[1.667,-4.2],[1,1.4],[-3.667,-4.2],
                                    [-3.333,4.2],[3.333,-1.4],[2.333,1.4],[-2.667,-1.4],
                                    [-3,1.4],[7,-1.4],[4.667,1.4],[-2.333,-4.2],
                                    [-2,4.2],[15,-1.4],[10.333,1.4],[-1.333,-1.4]
                                    ])
        self.laser_id = np.full((32,12),np.arange(32).reshape(-1,1).astype('int'))
        self.timing_offset = self.calc_timing_offsets()
        self.omega = self.Data_order[:,0]
        self.arg_omega = np.argsort(self.omega)
        self.lidar_reader = 0
        self.file_path = file_path
        self.load_reader()
        
    def load_reader(self):
        try:
            fpcap = open(self.file_path, 'rb')
            self.lidar_reader = dpkt.pcap.Reader(fpcap)
        except Exception as ex:
            print(str(ex))

    def read_uint32(self,data, idx):
        return data[idx] + data[idx+1]*256 + data[idx+2]*256*256 + data[idx+3]*256*256*256
    def read_firing_data(self,data):
        block_id = data[0] + data[1]*256
        azimuth = (data[2] + data[3] * 256) / 100 # degree
        firings = data[4:].reshape(32, 3) 
        distances = firings[:, 0] + firings[:, 1] * 256 # mm 
        intensities = firings[:, 2] # 0-255
        return distances, intensities, azimuth #(1,0)
        
    def calc_timing_offsets(self):
        timing_offsets = np.zeros((32,12))  # Init matrix
        # constants
        full_firing_cycle = 55.296  # μs
        single_firing = 2.304  # μs
        # compute timing offsets
        for x in range(12):
            for y in range(32):
                dataBlockIndex = (x * 2) + int((y / 16))
                dataPointIndex = y % 16
                timing_offsets[y][x] = (full_firing_cycle * dataBlockIndex) +(single_firing * dataPointIndex)
        return np.array(timing_offsets).T

    def calc_precise_azimuth(self,azimuth_per_block):

        # block_number: how many blocks are required to be processed 

        org_azi = azimuth_per_block.copy()
        precision_azimuth = []
        # iterate through each block
        for n in range(len(org_azi)): # n=0..11
            azimuth = org_azi.copy()
            try:
                # First, adjust for an Azimuth rollover from 359.99° to 0°
                if azimuth[n + 1] < azimuth[n]:
                    azimuth[n + 1] += 360

                # Determine the azimuth Gap between data blocks
                azimuth_gap = azimuth[n + 1] - azimuth[n]
            except:
                if azimuth[n] < azimuth[n-1]:
                    azimuth[n] += 360
                azimuth_gap = azimuth[n] - azimuth[n-1]

            factor = azimuth_gap / 32.
            k = np.arange(32)
            precise_azimuth = azimuth[n] + factor * k
            precision_azimuth.append(precise_azimuth)

        precision_azimuth = np.array(precision_azimuth).T
        return precision_azimuth # 32 * 12

    def calc_cart_coord(self,distances, azimuth):# distance: 12*32 azimuth: 12*32
        # convert deg to rad
        longitudes = self.omega * np.pi / 180.
        latitudes = azimuth * np.pi / 180.

        hypotenuses = distances * np.cos(longitudes)

        X = hypotenuses * np.sin(latitudes)
        Y = hypotenuses * np.cos(latitudes)
        Z = distances * np.sin(longitudes)
        return X, Y, Z
    

    def parse_one_packet(self,data):
        data = np.frombuffer(data, dtype=np.uint8).astype(np.uint32)
        blocks = data[0:1200].reshape(12, 100)
        # Timestamp = self.read_uint32(data[1200:1204],0)
        distances = []#12*32
        intensities = []#12*32
        azimuth_per_block = [] #(12,0)
        # iteratie through each block
        for i, blk in enumerate(blocks):
            dists, intens, angles = self.read_firing_data(blk)
            distances.append(dists) #12*32
            intensities.append(intens) #12*32
            azimuth_per_block.append(angles)

        azimuth_per_block = np.array(azimuth_per_block).T
        distances = 4/1000*np.array(distances).T # 32,12
        intensities = np.array(intensities).T # 32,12

        return distances,intensities, azimuth_per_block # 12*0
 
    

    def cal_angle_diff(self,advance_angle,lagging_angle):
        if advance_angle < lagging_angle:
            return advance_angle + 360 - lagging_angle
        else:
            return advance_angle - lagging_angle
    
    def frame_gen(self):

        while True:
            culmulative_azimuth = 0
            culmulative_azimuth_values = []
            culmulative_laser_ids = []
            culmulative_distances = []
            culmulative_intensities = []
            Td_map = np.zeros((32,1800))
            Intens_map = np.zeros((32,1800))

            while True:
                try:
                    ts,buf = next(self.lidar_reader)
                    eth = dpkt.ethernet.Ethernet(buf)
                except :
                    yield None
                        
                if eth.type == 2048: # for ipv4
                    if type(eth.data.data) == dpkt.udp.UDP:
                        data = eth.data.data.data
                        packet_status = eth.data.data.sport
                        if packet_status == 2368:
                            if len(data) != 1206:
                                continue
                            distances,intensities,azimuth_per_block = self.parse_one_packet(data)
                            azimuth = self.calc_precise_azimuth(azimuth_per_block) # 32 x 12
                            cur_azimuth = azimuth_per_block[-1]
                            culmulative_azimuth_values.append(azimuth)
                            culmulative_laser_ids.append(self.laser_id)
                            culmulative_distances.append(distances)
                            culmulative_intensities.append(intensities)
                            break 
                
            while True:
                try:
                    ts,buf = next(self.lidar_reader)
                    eth = dpkt.ethernet.Ethernet(buf)
                except :
                    yield None
                
                if eth.type == 2048:
                    if type(eth.data.data) == dpkt.udp.UDP:
                        data = eth.data.data.data
                        packet_status = eth.data.data.sport
                        if packet_status == 2368:
                            if len(data) != 1206:
                                culmulative_azimuth += diff 
                                continue
                            """
                            distances : (32,12)
                            intensities : (32,12)
                            azimuth_per_block : (12,0)
                            """
                            distances,intensities,azimuth_per_block = self.parse_one_packet(data)
                            azimuth = self.calc_precise_azimuth(azimuth_per_block) # 32 x 12
                            culmulative_azimuth_values.append(azimuth)
                            culmulative_laser_ids.append(self.laser_id)
                            culmulative_distances.append(distances)
                            culmulative_intensities.append(intensities)
                            diff = self.cal_angle_diff(azimuth_per_block[-1],cur_azimuth)
                            cur_azimuth = azimuth_per_block[-1]
                            culmulative_azimuth += diff 
                            
                            if culmulative_azimuth > 359.8: 
                                # print(len(culmulative_azimuth_values))
                                culmulative_azimuth_values = np.concatenate(culmulative_azimuth_values,axis = 1)
                                culmulative_azimuth_values += self.Data_order[:,1].reshape(-1,1)
                                culmulative_laser_ids = np.concatenate(culmulative_laser_ids,axis = 1).flatten()
                                culmulative_distances = np.concatenate(culmulative_distances,axis = 1).flatten()
                                culmulative_intensities = np.concatenate(culmulative_intensities,axis = 1).flatten()
                                culmulative_azimuth_inds = np.around(culmulative_azimuth_values/0.2).astype('int').flatten()
                                culmulative_azimuth_inds[culmulative_azimuth_inds > 1799] -= 1800
                                culmulative_azimuth_inds[culmulative_azimuth_inds < 0 ] += 1800
                                
                                Td_map[culmulative_laser_ids,culmulative_azimuth_inds] = culmulative_distances
                                Intens_map[culmulative_laser_ids,culmulative_azimuth_inds] = culmulative_intensities
                                # cur_azimuth = azimuth_per_block[-1]
                                # sys.stdout.write('\rProcessing Time: {} ms'.format(round((time.time() - a) * 1000,2)))
                                # sys.stdout.flush()
                                
                                yield Td_map[self.arg_omega,:],Intens_map[self.arg_omega,:] #32*1800
                                break

def get_image(Td_map,Inten_map):
    Image = np.concatenate([Td_map[:,:,np.newaxis],Inten_map[:,:,np.newaxis]],axis = -1)
    return Image

if __name__ == "__main__":
    # background_object = cv2.createBackgroundSubtractorMOG2(varThreshold=10)
    # background_object.setNMixtures(1)
    # background_object.setBackgroundRatio(0.5)
    thred_map = np.load(r'D:\Test\bck.npy')
    td_gen = TDmapLoader(r'D:\LiDAR_Data\MidTown\California\2021-12-8-18-0-0.pcap').frame_gen()
    db = Raster_DBSCAN(window_size=[5,15],eps = 1.5,min_samples= 10,Td_map_szie=thred_map.shape)
    
    vis = op3.visualization.Visualizer()
    vis.create_window()
    Td_map,Intens_map = next(td_gen)
    # img = get_image(Td_map,Intens_map)
    # fgmsk = background_object.apply(Td_map)
    
    Foreground_map = (Td_map < thred_map)
    source = get_pcd_colored(Td_map,Foreground_map)
    vis.add_geometry(source)
    i = 0
    while True:
        Td_map,Intens_map = next(td_gen)
        # img = get_image(Td_map,Intens_map)
        # fgmsk = background_object.apply(img,learningRate = 0.1)
        Foreground_map = Td_map < thred_map
        
        pcd = get_pcd_colored(Td_map,Foreground_map)
        source.points = pcd.points
        source.colors = pcd.colors
        # vis.capture_screen_image(f'D:\Test\Figs\{i}.png')
        vis.update_geometry(source)
        vis.poll_events()
        vis.update_renderer()   
        i += 1 
