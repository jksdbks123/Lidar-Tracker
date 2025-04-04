import sys
import os
interface_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', r'Utils'))
# Add Interface to sys.path
sys.path.insert(0, interface_path)
import numpy as np
from DDBSCAN import Raster_DBSCAN
import cv2

import time
import dpkt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
import socket

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
        self.state = None 
        self.apperance = None 
        self.rows = None
        self.cols = None # row, col inds in the Td-map 
        self.label_seq = [] # represented labels at each frame 
        self.mea_seq = []
        self.post_seq = []
        self.app_seq = []
        self.P_seq = []
        self.pred_state = [] # 0 : measured, 1: pred, 
        self.P = None
        self.missing_count = 0

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

def track_point_clouds(stop_event,mot,point_cloud_queue,result_queue,tracking_parameter_dict,tracking_param_update_event):
    start_tracking_time = time.time()
    while not stop_event.is_set():
        Td_map =  point_cloud_queue.get()
        # some steps
        
        if not mot.if_initialized:
            time_a = time.time()
            mot.initialization(Td_map)
            Tracking_pool = mot.Tracking_pool
            Labeling_map = mot.cur_Labeling_map
            time_b = time.time()
        else:
            if tracking_param_update_event.is_set():
                mot.db = Raster_DBSCAN(window_size=tracking_parameter_dict['win_size'],eps = tracking_parameter_dict['eps'], min_samples= tracking_parameter_dict['min_samples'],Td_map_szie=(32,1800))
                tracking_param_update_event.clear()
            time_a = time.time()
            mot.mot_tracking_step(Td_map)
            time_b = time.time()
            Tracking_pool = mot.Tracking_pool
            Labeling_map = mot.cur_Labeling_map

            
            if (time_b - start_tracking_time) > 120:
                 mot.Off_tracking_pool = {}
                 mot.Tracking_pool = {}
                 mot.Global_id = 0
                 start_tracking_time = time.time()
        result_queue.put((Tracking_pool,Labeling_map,Td_map,time_b - time_a))

    print('Terminated tracking process')

def load_pcap(file_path):
    try:
        fpcap = open(file_path, 'rb')
        eth_reader = dpkt.pcap.Reader(fpcap)
        return eth_reader
    except Exception as ex:
        print(str(ex))
        return None

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


def save_fore_pcd(Td_map,Labeling_map,save_path,frame_ind,Tracking_pool):
    Xs = []
    Ys = []
    Zs = []
    Labels = []
    for i in range(Td_map.shape[0]):
        longitudes = theta[i] * np.pi / 180
        latitudes = azimuths * np.pi / 180 
        hypotenuses = Td_map[i] * np.cos(longitudes)
        X = hypotenuses * np.sin(latitudes)
        Y = hypotenuses * np.cos(latitudes)
        Z = Td_map[i] * np.sin(longitudes)
        
        Valid_ind = Td_map[i] != 0 
        Xs.append(X[Valid_ind])
        Ys.append(Y[Valid_ind])
        Zs.append(Z[Valid_ind])
        Labels.append(Labeling_map[i][Valid_ind])

    Xs = np.concatenate(Xs)
    Ys = np.concatenate(Ys)
    Zs = np.concatenate(Zs)
    Labels = np.concatenate(Labels)   
    Labels_temp = Labels.copy()
    for glb_id in Tracking_pool:
        label_id = Tracking_pool[glb_id].label_seq[-1]
        if label_id != -1:
            Labels_temp[Labels == label_id] = glb_id

    pcd = np.c_[Xs,Ys,Zs,Labels_temp]
    pcd = pcd[pcd[:,3] != -1]
    
    np.save(os.path.join(save_path,f'{frame_ind}.npy'), pcd)

db_merge = DBSCAN(eps = 1.8, min_samples = 2)
def extract_xy(Labeling_map,Td_map):
        
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
def read_packets_offline(pcap_file_path):
    eth_reader = load_pcap(pcap_file_path)
    while True:
        # Simulate reading a packet from the Ethernet
        try:
            ts,buf = next(eth_reader)
        except StopIteration:
            return None
        try:
            eth = dpkt.ethernet.Ethernet(buf)
        except dpkt.dpkt.NeedData:
            continue
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

def read_packets_online(port,raw_data_queue):
    sock = socket.socket(socket.AF_INET, # Internet
                    socket.SOCK_DGRAM) # UDP
    sock.bind(('', port))     
    while True:
        data,addr = sock.recvfrom(1206)
        raw_data_queue.put_nowait((time.time(),data))
        print(raw_data_queue.qsize())

# Function to parse packets into point cloud data (Simulating Core 3)
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
            # flag = lane_drawer.if_rollover(azimuth_per_block,Initial_azimuth)
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

# def associate_detections(Tracking_pool,glb_id,state,app,P,next_label,mea_next):
    
#     Tracking_pool[glb_id].state = state
#     Tracking_pool[glb_id].apperance = app
#     Tracking_pool[glb_id].P = P
#     Tracking_pool[glb_id].label_seq.append(next_label)
#     Tracking_pool[glb_id].mea_seq.append(mea_next)
#     Tracking_pool[glb_id].post_seq.append(state)
#     Tracking_pool[glb_id].app_seq.append(app)
#     Tracking_pool[glb_id].missing_count = 0
#     Tracking_pool[glb_id].pred_state.append(0)

# def process_fails(Tracking_pool,Off_tracking_pool,glb_id,state_cur_,P_cur_,missing_thred):
#     Tracking_pool[glb_id].missing_count += 1
#     fail_condition1 = Tracking_pool[glb_id].missing_count > missing_thred
#     # dis = np.sqrt(np.sum(state_cur_[0][:2]**2))
#     if  fail_condition1:
#         Off_tracking_pool[glb_id] = Tracking_pool.pop(glb_id)
#     else:
#         Tracking_pool[glb_id].state = state_cur_
#         Tracking_pool[glb_id].P = P_cur_
#         Tracking_pool[glb_id].label_seq.append(-1)
#         Tracking_pool[glb_id].mea_seq.append(None)
#         Tracking_pool[glb_id].app_seq.append(Tracking_pool[glb_id].app_seq[-1])
#         Tracking_pool[glb_id].post_seq.append(state_cur_)
#         Tracking_pool[glb_id].pred_state.append(1)

# def create_new_detection(Tracking_pool,Global_id,P_init,state_init,app_init,label_init,mea_init,start_frame):

#     dis = np.sqrt(np.sum(state_init[0][:2]**2))

#     if dis > 0.1:
#         new_detection = detected_obj()
#         new_detection.glb_id = Global_id
#         new_detection.P = P_init
#         new_detection.state = state_init
#         new_detection.apperance = app_init
#         new_detection.label_seq.append(label_init)
#         new_detection.start_frame = start_frame
#         new_detection.mea_seq.append(mea_init)
#         new_detection.post_seq.append(state_init)
#         new_detection.app_seq.append(app_init)
#         new_detection.pred_state.append(0)
#         Tracking_pool[Global_id] = new_detection

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


def associate_detections(Tracking_pool, glb_id, state, app, P, next_label, mea_next):
    """Associate new detections with existing tracking objects."""
    obj = Tracking_pool[glb_id]  # Fetch from shared dict proxy
    obj.state = state
    obj.apperance = app
    obj.P = P
    obj.label_seq.append(next_label)
    obj.mea_seq.append(mea_next)
    obj.post_seq.append(state)
    obj.app_seq.append(app)
    obj.missing_count = 0
    Tracking_pool[glb_id] = obj  # Reassign to ensure changes persist in Manager.dict()


# (Tracking_pool,Off_tracking_pool,glb_id,state_cur_,P_cur_,missing_thred):
def process_fails(Tracking_pool, Off_tracking_pool, glb_id, state_cur_, P_cur_, missing_thred):
    """Handle lost detections by either updating state or removing them from tracking."""
    obj = Tracking_pool[glb_id]
    obj.missing_count += 1
    fail_condition1 = obj.missing_count > missing_thred

    if fail_condition1:
        Tracking_pool.pop(glb_id)  # Properly delete from shared dict
    else:
        obj.state = state_cur_
        obj.P = P_cur_
        obj.label_seq.append(-1)
        obj.mea_seq.append(None)
        obj.app_seq.append(-1)
        obj.post_seq.append(state_cur_)
        obj.pred_state.append(1)  # Update prediction state
        Tracking_pool[glb_id] = obj  # Reassign to update changes



def create_new_detection(Tracking_pool, Global_id, P_init, state_init, app_init, label_init, mea_init, start_frame):
    """Create a new detection and add it to the tracking pool."""
    dis = np.sqrt(np.sum(state_init[0][:2]**2))

    if dis > 0.5:
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
        Tracking_pool[Global_id] = new_detection  # Add to shared dict properly

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
    

def get_affinity_IoU(app_cur,app_next,unique_label_next,unique_label_cur,Labeling_map_cur,Labeling_map_next):
    # Union: only A or B 
    # Intersect : only A and B 
    # print(unique_label_next,unique_label_cur)
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
        # print(dis,unique_label_next,unique_label_cur)
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

