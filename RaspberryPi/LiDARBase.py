from Utils import *
from DDBSCAN import Raster_DBSCAN
import cv2
import sys
import os
from sklearn.cluster import DBSCAN
import numpy as np
import time
import socket
import gc
from queue import Empty, Full  # Standard exceptions
def safe_queue_get(q, timeout=5, default=None, queue_name="queue"):
    """
    Safely get an item from the queue with timeout.
    Returns `default` if queue is empty.
    """
    try:
        item = q.get(timeout=timeout)
        return item
    except Empty:
        print(f"[WARNING] {queue_name}: get() timed out after {timeout}s — queue is empty.")
        return default
    except Exception as e:
        print(f"[ERROR] {queue_name}: unexpected exception during get(): {e}")
        return default


def safe_queue_put(q, item, timeout=5, queue_name="queue"):
    """
    Safely put an item into the queue with timeout.
    Returns True if success, False if failed (queue full or error).
    """
    try:
        q.put(item, timeout=timeout)
        return True
    except Full:
        print(f"[WARNING] {queue_name}: put() timed out after {timeout}s — queue is full.")
        return False
    except Exception as e:
        print(f"[ERROR] {queue_name}: unexpected exception during put(): {e}")
        return False
    
np.random.seed(412)
color_map = (np.random.random((100,3)) * 255).astype(int)
color_map = np.concatenate([color_map,np.array([[255,255,255]]).astype(int)])
# red 0, blue 1
color_map_foreground = np.array([[255,0,0],[0,0,255]])
thred_map_index = np.arange(32*1800).reshape((32,1800))

class detected_obj():
    def __init__(self):
        self.glb_id = None
        self.start_frame = None
        self.missing_count = 0 # frame count of out of detection
        self.P = None
        self.state = None 
        self.apperance = None 
        self.label_seq = [] # represented labels at each frame 
        self.mea_seq = []
        self.post_seq = []
        self.app_seq = []

def calc_timing_offsets():
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

Data_order = np.array([[-25,1.4],[-1,-4.2],[-1.667,1.4],[-15.639,-1.4],
                            [-11.31,1.4],[0,-1.4],[-0.667,4.2],[-8.843,-1.4],
                            [-7.254,1.4],[0.333,-4.2],[-0.333,1.4],[-6.148,-1.4],
                            [-5.333,4.2],[1.333,-1.4],[0.667,4.2],[-4,-1.4],
                            [-4.667,1.4],[1.667,-4.2],[1,1.4],[-3.667,-4.2],
                            [-3.333,4.2],[3.333,-1.4],[2.333,1.4],[-2.667,-1.4],
                            [-3,1.4],[7,-1.4],[4.667,1.4],[-2.333,-4.2],
                            [-2,4.2],[15,-1.4],[10.333,1.4],[-1.333,-1.4]
                            ])
laser_id = np.full((32,12),np.arange(32).reshape(-1,1).astype('int'))
timing_offset = calc_timing_offsets()
omega = Data_order[:,0]
theta = np.sort(omega)
azimuths = np.arange(0,360,0.2)
arg_omega = np.argsort(omega)

"""
raw_data_queue: UDP packets from LiDAR snesor 
LidarVisualizer.point_cloud_queue: parsed point cloud frames 
"""
def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
def intersect_angle(A,B,C,D):
    vec_1 = np.array(A) - np.array(B)
    vec_2 = np.array(C) - np.array(D)
    norm_1 = np.linalg.norm(vec_1)
    norm_2 = np.linalg.norm(vec_2)
    if norm_1 == 0 or norm_2 == 0:
        return False
    cos_theta = np.dot(vec_1,vec_2) / (norm_1 * norm_2)
    # if almost vertical, then return true
    if np.abs(cos_theta) < 0.4: 
        return True
    return False

def line_segments_intersect(seg1_start, seg1_end, seg2_start, seg2_end):
    # seg : (x,y)
    """Returns True if line segments seg1 and seg2 intersect."""
    flag_1 = ccw(seg1_start, seg2_start, seg2_end) != ccw(seg1_end, seg2_start, seg2_end) and ccw(seg1_start, seg1_end, seg2_start) != ccw(seg1_start, seg1_end, seg2_end)
    flag_2 = intersect_angle(seg1_start,seg1_end,seg2_start,seg2_end)
    return flag_1 & flag_2

def track_point_clouds(stop_event,mot,point_cloud_queue,tracking_parameter_dict,tracking_param_update_event,background_update_event, thred_map_dict,bar_drawer,memory_clear_time = 10):
    start_tracking_time = time.time()
    try:
        while not stop_event.is_set():

            Td_map = safe_queue_get(point_cloud_queue, timeout=5, default=None, queue_name="point_cloud_queue")

            # some steps
            
            if not mot.if_initialized:
                time_a = time.time()
                mot.initialization(Td_map)
                time_b = time.time()
            else:
                if tracking_param_update_event.is_set():
                    mot.db = Raster_DBSCAN(window_size=tracking_parameter_dict['win_size'],eps = tracking_parameter_dict['eps'], min_samples= tracking_parameter_dict['min_samples'],Td_map_szie=(32,1800))
                    tracking_param_update_event.clear()
                if background_update_event.is_set():
                    mot.thred_map = thred_map_dict['thred_map']
                    background_update_event.clear()
                time_a = time.time()
                mot.mot_tracking_step(Td_map)
                time_b = time.time()
                # timely clear memory
                # if (time_b - start_tracking_time) > memory_clear_time:
                #     mot.Off_tracking_pool.clear()
                #     mot.Tracking_pool.clear() 
                #     gc.collect()
                #     mot.Global_id = 0
                #     start_tracking_time = time.time()
                #     print('Memory Cleared at {}'.format(start_tracking_time))
                tracking_dic = mot.Tracking_pool
                sys.stdout.write(f'\rData Processing Speed (ms): {mot.clustering_time:.3f}, {mot.bf_time:.3f}, {mot.association_time:.3f},{(time_b - time_a)*1000:.3f},Tracking{len(tracking_dic.keys()):.1f}')
                sys.stdout.flush()
    except Exception as ex:
        print(str(ex), 'Error in tracking process')
    print('Terminated tracking process')

def load_pcap(file_path):
    try:
        fpcap = open(file_path, 'rb')
        eth_reader = dpkt.pcap.Reader(fpcap)
    except Exception as ex:
        print(str(ex))
        return None
    return eth_reader

def parse_one_packet(data):
    data = np.frombuffer(data, dtype=np.uint8).astype(np.uint32)
    blocks = data[0:1200].reshape(12, 100)
    Timestamp = read_uint32(data[1200:1204],0)
    distances = []# 12*32
    intensities = []# 12*32
    azimuth_per_block = [] # (12,0)
    # iteratie through each block
    for i, blk in enumerate(blocks):
        dists, intens, angles = read_firing_data(blk)
        distances.append(dists) #12*32
        intensities.append(intens) #12*32
        azimuth_per_block.append(angles)

    azimuth_per_block = np.array(azimuth_per_block).T
    distances = 4/1000*np.array(distances).T # 32,12
    intensities = np.array(intensities).T # 32,12

    return distances,intensities, azimuth_per_block, Timestamp # 12*0

def read_uint32(data, idx):
    return data[idx] + data[idx+1]*256 + data[idx+2]*256*256 + data[idx+3]*256*256*256

def read_firing_data(data):
    block_id = data[0] + data[1]*256
    azimuth = (data[2] + data[3] * 256) / 100 # degree
    firings = data[4:].reshape(32, 3) 
    distances = firings[:, 0] + firings[:, 1] * 256 # mm 
    intensities = firings[:, 2] # 0-255
    return distances, intensities, azimuth #(1,0)
    


def calc_precise_azimuth(azimuth_per_block):

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

def calc_cart_coord(distances, azimuth):# distance: 12*32 azimuth: 12*32
    # convert deg to rad
    longitudes = omega * np.pi / 180.
    latitudes = azimuth * np.pi / 180.

    hypotenuses = distances * np.cos(longitudes)

    X = hypotenuses * np.sin(latitudes)
    Y = hypotenuses * np.cos(latitudes)
    Z = distances * np.sin(longitudes)
    return X, Y, Z

def get_ordinary_point_cloud(Td_map,vertical_limits):
    point_cloud_data = get_pcd_uncolored(Td_map,vertical_limits)
    return point_cloud_data,None,None

def get_foreground_point_cloud(thred_map,bck_radius,Td_map,vertical_limits):
    Foreground_map = ~(np.abs(Td_map - thred_map) <= bck_radius).any(axis = 0)
    Foreground_map = Foreground_map.astype(int)
    point_cloud_data,labels = get_pcd_colored(Td_map,Foreground_map,vertical_limits)                    

    return point_cloud_data,labels,None

def get_pcd_tracking(Td_map,Labeling_map,Tracking_pool,vertical_limits):
    Xs = []
    Ys = []

    Labels = []
    for i in range(vertical_limits[0],vertical_limits[1]):
        longitudes = theta[i] * np.pi / 180
        latitudes = azimuths * np.pi / 180 
        hypotenuses = Td_map[i] * np.cos(longitudes)
        X = hypotenuses * np.sin(latitudes)
        Y = hypotenuses * np.cos(latitudes)
        Valid_ind = Td_map[i] != 0 
        Xs.append(X[Valid_ind])
        Ys.append(Y[Valid_ind])
        Labels.append(Labeling_map[i][Valid_ind])

    Xs = np.concatenate(Xs)
    Ys = np.concatenate(Ys)
        
    Labels = np.concatenate(Labels)
    # Colors = np.full((len(Labels),3),np.array([[153,153,153]])/256)
    for key in Tracking_pool:
        label_cur_frame = Tracking_pool[key].label_seq[-1]
        if label_cur_frame != -1:
            # Colors[Labels == label_cur_frame] = color_map[key%len(color_map)]
            Labels[Labels == label_cur_frame] = key
            
    XYZ = np.c_[Xs,Ys]
    return XYZ,Labels

def get_pcd_uncolored(Td_map,vertical_limits):

    Xs = []
    Ys = []
    for i in range(vertical_limits[0],vertical_limits[1]):
        longitudes = theta[i]*np.pi / 180
        latitudes = azimuths * np.pi / 180 
        hypotenuses = Td_map[i] * np.cos(longitudes)
        X = hypotenuses * np.sin(latitudes)
        Y = hypotenuses * np.cos(latitudes)
        Xs.append(X)
        Ys.append(Y)

    Xs = np.concatenate(Xs)
    Ys = np.concatenate(Ys)
    
    XYZ = np.c_[Xs,Ys]
    XYZ = XYZ[(XYZ[:,0] != 0)&(XYZ[:,1] != 0)]
    return XYZ

def get_pcd_colored(Td_map,Labeling_map,vertical_limits):

    Xs = []
    Ys = []
    Labels = []
    for i in range(vertical_limits[0],vertical_limits[1]):
        longitudes = theta[i]*np.pi / 180
        latitudes = azimuths * np.pi / 180 
        hypotenuses = Td_map[i] * np.cos(longitudes)
        X = hypotenuses * np.sin(latitudes)
        Y = hypotenuses * np.cos(latitudes)
        Xs.append(X)
        Ys.append(Y)
        Labels.append(Labeling_map[i])

    Xs = np.concatenate(Xs)
    Ys = np.concatenate(Ys)
    Labels = np.concatenate(Labels)
    XYZ = np.c_[Xs,Ys]
    Valid_ind = (XYZ[:,0] != 0)&(XYZ[:,1] != 0)
    Labels = Labels[Valid_ind]
    XYZ = XYZ[Valid_ind]

    return XYZ,Labels    
 
def get_pcd_colored_laser_ind(Td_map,Labeling_map,vertical_limits,thred_map_index):

    Xs = []
    Ys = []
    Labels = []
    LaserInds = []
    for i in range(vertical_limits[0],vertical_limits[1]):
        
        longitudes = theta[i]*np.pi / 180
        latitudes = azimuths * np.pi / 180 
        hypotenuses = Td_map[i] * np.cos(longitudes)
        X = hypotenuses * np.sin(latitudes)
        Y = hypotenuses * np.cos(latitudes)
        Xs.append(X)
        Ys.append(Y)
        Labels.append(Labeling_map[i])
        LaserInds.append(thred_map_index[i])

    Xs = np.concatenate(Xs)
    Ys = np.concatenate(Ys)
    Labels = np.concatenate(Labels)
    LaserInds = np.concatenate(LaserInds)
    XYZ = np.c_[Xs,Ys]
    Valid_ind = (XYZ[:,0] != 0)&(XYZ[:,1] != 0)
    Labels = Labels[Valid_ind]
    LaserInds = LaserInds[Valid_ind]
    XYZ = XYZ[Valid_ind]

    return XYZ,Labels,LaserInds

def get_static_bck_points(thred_map,vertical_limits):

    bck_points_total = []
    for i in range(thred_map.shape[0]):
        Labeling_map = thred_map[i] > 0
        bck_points,Labels = get_pcd_colored(thred_map[i],Labeling_map,vertical_limits)
        bck_points_total.append(bck_points[Labels])
    bck_points_total = np.concatenate(bck_points_total)

    return bck_points_total

# # # Simulated function to continuously read packets (Simulating Core 2)
def read_packets_offline(raw_data_queue,pcap_file_path):
    eth_reader = load_pcap(pcap_file_path)
    while True:
        # Simulate reading a packet from the Ethernet
        try:
            ts,buf = next(eth_reader)
            eth = dpkt.ethernet.Ethernet(buf)
        except:
            # when it's empty, reload current pcap
            eth_reader = load_pcap(pcap_file_path)
        if eth.type == 2048: # for ipv4
            if (type(eth.data.data) == dpkt.udp.UDP):# for ipv4
                data = eth.data.data.data
                packet_status = eth.data.data.sport
                if packet_status == 2368:
                    if len(data) != 1206:
                        continue
            # raw_packet = np.random.rand(20000,2) * 600  # Placeholder for actual packet data
                    raw_data_queue.put((ts,data),timeout = 0.5)

# def read_packets_online(port,raw_data_queue):

#     sock = socket.socket(socket.AF_INET, # Internet
#                                 socket.SOCK_DGRAM) # UDP
#     sock.bind(('', port))     
#     while True:
#         data,addr = sock.recvfrom(1206)
#         raw_data_queue.put((time.time(),data))

# def read_packets_online(port, raw_data_queue):
    
#     """Continuously reads packets but behaves differently based on mode."""
#     sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     sock.bind(('', port))
#     sock.settimeout(5)  # Prevent indefinite blocking

#     while True:
#         try:
#             data, addr = sock.recvfrom(2048)  # Receive data from LiDAR
#             # print(f"[DEBUG] Received {len(data)} bytes from {addr}")
#             raw_data_queue.put_nowait((time.time(),data))
#         except socket.timeout:
#             print("[WARNING] No data received in 5 seconds. LiDAR may have stopped sending.")
#         except Exception as e:
#             print(f"[ERROR] Socket error: {e}")
#             break  # Exit if an unrecoverable error occurs

    
def read_packets_online(port, raw_data_queue):
    """Continuously reads packets and logs receiving rate every 5 seconds."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', port))
    sock.settimeout(1)  # Shorter timeout allows more responsive counting

    # packet_count = 0
    # interval_start = time.time()

    while True:
        try:
            data, addr = sock.recvfrom(2048)
            # print(f"[DEBUG] Received {len(data)} bytes from {addr}")
            if len(data) != 1206:
                print(f"[WARNING] Received packet of unexpected length: {len(data)} bytes.")
                continue
            safe_queue_put(raw_data_queue, (time.time(), data), timeout=0.5, queue_name="raw_data_queue")
            # raw_data_queue.put((time.time(), data),timeout = 0.5)
            # packet_count += 1
        except socket.timeout:
            # Not an error, just no data for 1 second
            pass
        except Exception as e:
            print(f"[ERROR] Socket error: {e}")
            break

        # # Every 10 seconds, print the packet rate
        # if time.time() - interval_start >= 5:
        #     print(f"[INFO] Received {packet_count} packets in the last 5 seconds.")
        #     if packet_count == 0:
        #         print("[WARNING] No packets received! Possible sensor failure or network issue.")
        #     packet_count = 0
        #     interval_start = time.time()

import csv

class TimestampLogger:
    def __init__(self, log_path="./timestamp_log.csv"):
        self.log_path = log_path
        self.first_time = time.time()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["WallTime", "TimestampFromPacket", "ElapsedSinceStart(s)"])

    def log(self, packet_timestamp):
        now = time.time()
        elapsed = now - self.first_time
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([now, packet_timestamp, elapsed])

def parse_packets(raw_data_queue, point_cloud_queue,background_point_cloud_queue = None, background_point_copy_event = None):
    C_1 = np.int64(3599999999)
    C_2 = np.int64(C_1 / 2)
    culmulative_azimuth_values = []
    culmulative_laser_ids = []
    culmulative_distances = []
    # culmulative_intensities = []
    Td_map = np.zeros((32,1800))
    # Intens_map = np.zeros((32,1800))
    next_ts = 0
    # print('Parse Packet Process Started 111')
    ts,raw_packet = safe_queue_get(raw_data_queue, timeout=5, default=(0, None), queue_name="raw_data_queue")
    # ts,raw_packet = raw_data_queue.get()
    # print('Parse Packet Process Started')
    distances,intensities,azimuth_per_block,Timestamp = parse_one_packet(raw_packet)
    # print(Timestamp)
    next_ts = Timestamp + 100_000
    azimuth = calc_precise_azimuth(azimuth_per_block) # 32 x 12
    culmulative_azimuth_values.append(azimuth)
    culmulative_laser_ids.append(laser_id)
    culmulative_distances.append(distances)

    while True:
        while True:
            ts,raw_packet = safe_queue_get(raw_data_queue, timeout=5, default=(0, None), queue_name="raw_data_queue")
            # print("[Parsing] Get for new packets...")
            # Placeholder for parsing logic; here we just pass the data through
            distances,intensities,azimuth_per_block,Timestamp = parse_one_packet(raw_packet)
            azimuth = calc_precise_azimuth(azimuth_per_block) # 32 x 12
            # print(Timestamp, next_ts)
            new_frame_flag = False
            
            if np.int64(Timestamp) - np.int64(next_ts) > np.int64(100_000):
                # it's a roll over
                if Timestamp < C_2:
                    if Timestamp > next_ts:
                        print(f"[Parsing] packet timestamp{Timestamp} next_ts{next_ts} diff{Timestamp - next_ts}")
                        new_frame_flag = True
            else:
                if Timestamp > next_ts:
                    new_frame_flag = True

            if new_frame_flag:
                # print(f"[Parsing] packet timestamp{Timestamp} next_ts{next_ts} diff{Timestamp - next_ts}")
                if len(culmulative_azimuth_values) > 0:
                    
                    culmulative_azimuth_values = np.concatenate(culmulative_azimuth_values,axis = 1)
                    culmulative_azimuth_values += Data_order[:,1].reshape(-1,1)
                    culmulative_laser_ids = np.concatenate(culmulative_laser_ids,axis = 1).flatten()
                    culmulative_distances = np.concatenate(culmulative_distances,axis = 1).flatten()
                    # culmulative_intensities = np.concatenate(culmulative_intensities,axis = 1).flatten()
                    culmulative_azimuth_inds = np.around(culmulative_azimuth_values/0.2).astype('int').flatten()
                    culmulative_azimuth_inds[(culmulative_azimuth_inds<0)|(culmulative_azimuth_inds>1799)] = culmulative_azimuth_inds[(culmulative_azimuth_inds<0)|(culmulative_azimuth_inds>1799)] % 1799

                    Td_map[culmulative_laser_ids,culmulative_azimuth_inds] = culmulative_distances
                    # Intens_map[culmulative_laser_ids,culmulative_azimuth_inds] = culmulative_intensities
                    safe_queue_put(point_cloud_queue, Td_map[arg_omega,:], timeout=0.5, queue_name="point_cloud_queue")
                    # print("[Parsing] Put for new point cloud...")
                    # point_cloud_queue.put(Td_map[arg_omega,:],timeout = 0.5) #32*1800
                    if  background_point_copy_event is not None:
                        if background_point_copy_event.is_set():
                            # background_point_cloud_queue.put(Td_map[arg_omega,:],timeout = 0.5)
                            safe_queue_put(background_point_cloud_queue, Td_map[arg_omega,:], timeout=0.5, queue_name="background_point_cloud_queue")

                else:
                    # point_cloud_queue.put(Td_map) #32*1800
                    safe_queue_put(point_cloud_queue, Td_map, timeout=0.5, queue_name="point_cloud_queue")
                    # print("[Parsing] Put for new point cloud...")
                    
                    if  background_point_copy_event is not None:
                        if background_point_copy_event.is_set():
                            # background_point_cloud_queue.put(Td_map,timeout = 0.5)
                            safe_queue_put(background_point_cloud_queue, Td_map, timeout=0.5, queue_name="background_point_cloud_queue")


                culmulative_azimuth_values = []
                culmulative_laser_ids = []
                culmulative_distances = []
                # culmulative_intensities = []

                Td_map = np.zeros((32,1800))
                # Intens_map = np.zeros((32,1800))
                next_ts += np.int64(100_000) 
                if next_ts > C_1:
                    next_ts -= C_1
                break
            else:
                culmulative_azimuth_values.append(azimuth)
                culmulative_laser_ids.append(laser_id)
                culmulative_distances.append(distances)
                # culmulative_intensities.append(intensities)

def associate_detections(Tracking_pool,glb_id,state,app,P,next_label,mea_next):
    
    Tracking_pool[glb_id].state = state
    Tracking_pool[glb_id].apperance = app
    Tracking_pool[glb_id].P = P
    Tracking_pool[glb_id].label_seq.append(next_label)
    Tracking_pool[glb_id].mea_seq.append(mea_next)
    Tracking_pool[glb_id].post_seq.append(state)
    Tracking_pool[glb_id].app_seq.append(app)
    Tracking_pool[glb_id].missing_count = 0

def process_fails(Tracking_pool,Off_tracking_pool,glb_id,state_cur_,P_cur_,missing_thred):
    Tracking_pool[glb_id].missing_count += 1
    fail_condition1 = Tracking_pool[glb_id].missing_count > missing_thred
    # dis = np.sqrt(np.sum(state_cur_[0][:2]**2))
    if  fail_condition1:
        # Off_tracking_pool[glb_id] = Tracking_pool.pop(glb_id)
        Tracking_pool.pop(glb_id)
        pass
    else:
        Tracking_pool[glb_id].state = state_cur_
        Tracking_pool[glb_id].P = P_cur_
        Tracking_pool[glb_id].label_seq.append(-1)
        Tracking_pool[glb_id].mea_seq.append(None)
        Tracking_pool[glb_id].app_seq.append(-1)
        Tracking_pool[glb_id].post_seq.append(state_cur_)

def state_predict(A,Q,state,P):
    """
    state: s_k-1, (n x 10 x 1)
    Cov: P_k-1 (n x 10 x 10)
    """
    # print(A.shape,state.shape)
    state_ = np.matmul(A,state)
    
    P_ = np.matmul(np.matmul(A,P),A.transpose()) + Q
    return state_,P_

def state_update(A,H,state_,P_,R,mea):
    """
    mea: m_k (m x 5 x 1)
    
    """
    K = np.matmul(np.matmul(P_,H.transpose()),np.linalg.inv(np.matmul(np.matmul(H,P_),H.transpose()) + R))
    P = np.matmul((np.eye(A.shape[0]) - np.matmul(K,H)),P_)
    residual = mea - np.matmul(H,state_) # n x 5 x 1
    state = state_ + np.matmul(K,residual)
    
    return state, P 

def create_new_detection(Tracking_pool,Global_id,P_init,state_init,app_init,label_init,mea_init,start_frame):

    dis = np.sqrt(np.sum(state_init[0][:2]**2))

    if dis > 10:
        new_detection = detected_obj()
        new_detection.glb_id = Global_id
        new_detection.P = P_init
        new_detection.state = state_init
        new_detection.apperance = app_init
        new_detection.label_seq.append(label_init)
        new_detection.start_frame = start_frame
        new_detection.mea_seq.append(mea_init)
        new_detection.post_seq.append(state_init)
        new_detection.app_seq.append(app_init)
        Tracking_pool[Global_id] = new_detection

def if_bck(rows,cols,Td_map,Plane_model):
    # check if an object is background
    # car: 2.6m 
    td_freq_map = Td_map
    longitudes = theta[rows]*np.pi / 180
    latitudes = azimuths[cols] * np.pi / 180 
    hypotenuses = td_freq_map[rows,cols] * np.cos(longitudes)
    X = hypotenuses * np.sin(latitudes)
    Y = hypotenuses * np.cos(latitudes)
    Z = td_freq_map[rows,cols] * np.sin(longitudes)
    Height_from_ground = Plane_model[0] * X + Plane_model[1] * Y  + Plane_model[2] * Z + Plane_model[3] 
    Max_Height = Height_from_ground.max()
    if (Max_Height > 3)|(Max_Height < 0.3):
        return True
    else:
        return False
    
def get_appearance_features(rows,cols,Td_map): #obtain length height and width
    
    # 1.dis 2. point cnt 3.Dir(x) 4.dir(y) 5.Height 6.Len 7.Width 8.2Darea

    td_freq_map = Td_map
    dis = td_freq_map[rows,cols]
    longitudes = theta[rows]*np.pi / 180
    latitudes = azimuths[cols] * np.pi / 180 
    hypotenuses = dis * np.cos(longitudes)
    X = hypotenuses * np.sin(latitudes)
    Y = hypotenuses * np.cos(latitudes)
    Z = dis * np.sin(longitudes)
    points = np.array([X,Y]).T
    points_num = len(points)
    rect = cv2.minAreaRect(points.astype('float32'))
    box = cv2.boxPoints(rect)
    # box = cv2.boxPoints(rect)
    b1 = np.sqrt(np.sum((box[1] - box[0])**2))
    b2 = np.sqrt(np.sum((box[2] - box[1])**2))
    length = b1
    width = b2
    dir_vec = box[1] - box[0]
    if b1 < b2:
        length = b2
        width = b1
        dir_vec = box[2] - box[1]
    modility = np.sqrt(np.sum(dir_vec**2))
    if modility == 0:
        dir_vec = np.array([0,0])
    else:
        dir_vec = dir_vec/modility
    height = Z.max() - Z.min()
    area = b1 * b2
    vec = np.array([points_num,dir_vec[0],dir_vec[1],height,length,width,area,dis.mean()]).reshape(-1,1)
    # vec = np.full((2,8,1),vec) # status vec for two representative points 
    return vec #1 x 8 x 1  

def get_representative_point(ref_rows,ref_cols,Td_map): 
    td_freq_map = Td_map
    longitudes = theta[ref_rows]*np.pi / 180
    latitudes = azimuths[ref_cols] * np.pi / 180 
    hypotenuses = td_freq_map[ref_rows,ref_cols] * np.cos(longitudes)
    X = hypotenuses * np.sin(latitudes)
    Y = hypotenuses * np.cos(latitudes)
    # Z = td_freq_map[ref_rows,ref_cols] * np.sin(longitudes)
    
    return np.array([
        [X[0],Y[0]],
        [X[1],Y[1]]
    ]).reshape(2,2,1) # n_repr x xy_dim x 1 

def get_xy_set(new_uni_labels,Labeling_map,Td_map,if_app):
    xy_set = [] # xy position and apperance features
    if if_app:
        apperance_set = []
    for label in new_uni_labels:
        rows,cols = np.where(Labeling_map == label)
        rows_temp,cols_temp = rows.copy(),cols.copy()
        sort_ind = np.argsort(cols)
        refer_cols = cols[sort_ind[[0,-1]]]
        # this is being said, the first place is for less azimuth id 
        refer_rows = rows[sort_ind[[0,-1]]]
        if np.abs(refer_cols[0] - refer_cols[1]) >= 900:
            cols[cols <= 900] += 1800
            sort_ind = np.argsort(cols)
            refer_cols = cols[sort_ind[[0,-1]]]
            refer_cols[refer_cols >= 1800] -= 1800
            refer_rows = rows[sort_ind[[0,-1]]]
        if if_app:
            apperance = get_appearance_features(rows_temp,cols_temp,Td_map)
            apperance_set.append(apperance)
        xy = get_representative_point(refer_rows,refer_cols,Td_map) # x,y vec for two representatives 
        xy_set.append(xy)
    xy_set = np.array(xy_set)
    if if_app:
        apperance_set = np.array(apperance_set)
        return xy_set,apperance_set
    else:
        return xy_set
    
db_merge = DBSCAN(eps = 1.8, min_samples = 2)

def extract_xy(Labeling_map,Td_map):
        
    # Plane_model is a 1 x 4 array representing a,b,c,d in ax + by + cz + d = 0 
    new_uni_labels = np.unique(Labeling_map[Labeling_map != -1])
    xy_set = get_xy_set(new_uni_labels,Labeling_map,Td_map,False)
    if len(xy_set) > 0:
        total_labels = np.concatenate([new_uni_labels,new_uni_labels])
        edge_points = np.concatenate([xy_set[:,1,:,0],xy_set[:,0,:,0]])
        merge_labels = db_merge.fit_predict(edge_points)
        unique_merge_labels = np.unique(merge_labels[merge_labels != -1])
        merge_pairs = [total_labels[merge_labels == l] for l in unique_merge_labels]
        for p in merge_pairs:
            merging_p = np.unique(p)
            if len(merging_p) > 1:
                for i in range(1,len(merging_p)):
                    Labeling_map[Labeling_map == merging_p[i]] = merging_p[0]
        new_uni_labels = np.unique(Labeling_map[Labeling_map != -1])

        xy_set,apperance_set = get_xy_set(new_uni_labels,Labeling_map,Td_map,True)
        return xy_set,apperance_set,new_uni_labels,Labeling_map
    else:
        return xy_set,[],new_uni_labels,Labeling_map

def get_affinity_IoU(app_cur,app_next,unique_label_next,unique_label_cur,Labeling_map_cur,Labeling_map_next):
    # Union: only A or B 
    # Intersect : only A and B 
    
    Fore_next = Labeling_map_next != -1
    Fore_cur = Labeling_map_cur != -1
    Union = Fore_cur|Fore_next
    Intersect = Fore_cur & Fore_next
    Union[Intersect] = False
    
    labels_next_union,labels_cur_union = Labeling_map_next[Union],Labeling_map_cur[Union]
    pairs_union,counts_union = np.unique(np.array([labels_cur_union,labels_next_union]).T,return_counts=True,axis = 0)
    
    labels_next_intersect,labels_cur_intersect = Labeling_map_next[Intersect],Labeling_map_cur[Intersect]
    pairs_intersect,counts_intersect = np.unique(np.array([labels_cur_intersect,labels_next_intersect]).T,return_counts=True,axis = 0)

    IoU_matrix = np.zeros((unique_label_cur.shape[0],unique_label_next.shape[0]))
    dis_matrix = np.ones((unique_label_cur.shape[0],unique_label_next.shape[0]))

    for i,pair in enumerate(pairs_intersect):
        cur_label,next_label = pair[0],pair[1]
        Intersection_p = counts_intersect[i]
        A_p = counts_union[(pairs_union[:,0] == cur_label)]
        if A_p.size == 0:
            A_p = 0
        B_p = counts_union[(pairs_union[:,1] == next_label)]
        if B_p.size == 0:
            B_p = 0
        Union_p = Intersection_p + A_p + B_p
        cur_ind = unique_label_cur == cur_label
        next_ind = unique_label_next == next_label
        IoU_matrix[cur_ind,next_ind] = Intersection_p/Union_p
        dis = np.abs(app_next[next_ind,-1,0] - app_cur[cur_ind,-1,0])
        if dis > 2:
            IoU_matrix[cur_ind,next_ind] = 0
            continue
        dis_matrix[cur_ind,next_ind] = dis/2

    return 0.7*IoU_matrix + 0.3*(1 - dis_matrix)
 
def get_affinity_kalman(failed_tracked_ind,new_detection_ind,state_cur_,mea_next,P_cur_):
    State_affinity =  1.5*np.ones((len(failed_tracked_ind),len(new_detection_ind)))
    for i,glb_ind in enumerate(failed_tracked_ind):
        state_pred = state_cur_[glb_ind].copy().reshape(2,-1)[:,:2]
        for j,label_ind in enumerate(new_detection_ind):
            mea = mea_next[label_ind].copy().reshape(2,-1)
            for k in range(state_pred.shape[0]):
                mal_dis = distance.mahalanobis(mea[k],state_pred[k],np.linalg.inv(P_cur_[i][k][:2,:2]))
                if mal_dis < State_affinity[i,j]:
                    State_affinity[i,j] = mal_dis
    return State_affinity

def linear_assignment(State_affinity):

    associated_ind_cur,associated_ind_next = [],[]
    associated_ind_cur_extend_,associated_ind_next_extend_= linear_sum_assignment(State_affinity,maximize = True)
    for i in range(len(associated_ind_cur_extend_)):
        if State_affinity[associated_ind_cur_extend_[i],associated_ind_next_extend_[i]] != 0:
            associated_ind_cur.append(associated_ind_cur_extend_[i])
            associated_ind_next.append(associated_ind_next_extend_[i])
    associated_ind_cur,associated_ind_next = np.array(associated_ind_cur),np.array(associated_ind_next)

    return associated_ind_cur,associated_ind_next

def linear_assignment_kalman(State_affinity):

    associated_ind_cur,associated_ind_next = [],[]
    associated_ind_cur_extend_,associated_ind_next_extend_= linear_sum_assignment(State_affinity,maximize = False)
    for i in range(len(associated_ind_cur_extend_)):
        if State_affinity[associated_ind_cur_extend_[i],associated_ind_next_extend_[i]] < 1.5:
            associated_ind_cur.append(associated_ind_cur_extend_[i])
            associated_ind_next.append(associated_ind_next_extend_[i])
    associated_ind_cur,associated_ind_next = np.array(associated_ind_cur),np.array(associated_ind_next)

    return associated_ind_cur,associated_ind_next

