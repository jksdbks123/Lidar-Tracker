from tqdm import tqdm
import dpkt
import numpy as np
import open3d as op3 
import pandas as pd
import os
class RansacCollector():
    def __init__(self,pcap_path,frames_set): 
        """
        frames_set -> frame indexes to use in the development of td map
        td_maps -> 1800 * 32 * ts
        thred_map -> thredshold map
        """
        
        
        self.frames_set = frames_set
        self.pcap_path = pcap_path
        self.td_maps = None
        self.thred_map = None
        if 'Output File' not in os.listdir(os.getcwd()):
            os.mkdir('Output File')
        self.output_path = os.path.join(os.getcwd(),'Output File')
        
    def gen_tdmap(self):
        
        lidar_reader = TDmapLoader(self.pcap_path)
        frame_gen = lidar_reader.frame_gen()
        td_maps = [] 
        print('Loading pcap...')
        for i in tqdm(range(np.max(self.frames_set)+1)):
            one_frame = next(frame_gen)
            td_maps.append(one_frame)
        aggregated_maps = np.array(td_maps) # 1800 * 32 * len(ts)
        self.td_maps = aggregated_maps
        # aggregated_maps_sorted = aggregated_maps[:,:,lidar_reader.Data_order[:,0].argsort()] 
    
    def gen_thredmap(self):
        if self.td_maps is None:
            return 0
        aggregated_maps_temp = self.td_maps[self.frames_set,:,:].copy()
        for i in tqdm(range(1800)):
            for j in range(32):
                ts_b = aggregated_maps_temp[:,i,j].copy()
                ts_b[ts_b == 0] = 200
                aggregated_maps_temp[:,i,j] = ts_b
        threshold_map = np.zeros((1800,32))
        print('Generating Threshold Map')
        for i in tqdm(range(1800)):
            for j in range(32):
                t_s = aggregated_maps_temp[:,i,j].copy()
                threshold_value = self.get_thred(t_s,step=0.1)
                threshold_map[i,j] = threshold_value
        self.thred_map = threshold_map
        
    def get_thred(self,ts,d = 1.4,thred_s = 0.6,N = 20,delta_thred = 1e-3,step = 0.1):# Ransac Para
        samples = []
        ts = ts[ts!=200]
        if len(ts) == 0:
            return 200
        flag = True                             
        for i in range(N):
            sample = np.random.choice(ts,replace=False)
            samples.append(sample)
            if len(ts[(ts > sample - d)&(ts < sample + d)])/len(ts) > thred_s:
                flag = False
                break
        if flag:
            return 200
        cur_thred = sample
        while True:
            next_thred = cur_thred - step
            if (len(ts[ts > next_thred])/len(ts) - len(ts[ts > cur_thred])/len(ts)) < delta_thred:
                break
            cur_thred = next_thred
        return next_thred
    
    def gen_pcdseq(self,frame_index_list): # gen_pcdseq is to generate pcd sequence, background points are represented by red and the targets are labeled by blue
        pcds_dir = os.path.join(self.output_path,'PcdSequence')
        if 'PcdSequence' not in os.listdir(self.output_path):
            os.mkdir(pcds_dir)
        print('Saving Pcd Sequence')
        for i in tqdm(frame_index_list):
            pcd = self.gen_pcd(i)
            op3.io.write_point_cloud(pcds_dir+"/{}.pcd".format(i), pcd)
        
                
    def save_tdmap(self):
        np.save(os.path.join(self.output_path,'thred_map.npy'),self.thred_map)
        print('Thred Map Saved at',os.path.join(self.output_path ))
    
    def gen_pcd(self,frame_index):
        theta = np.array([[-25,1.4],[-1,-4.2],[-1.667,1.4],[-15.639,-1.4],
                            [-11.31,1.4],[0,-1.4],[-0.667,4.2],[-8.843,-1.4],
                            [-7.254,1.4],[0.333,-4.2],[-0.333,1.4],[-6.148,-1.4],
                            [-5.333,4.2],[1.333,-1.4],[0.667,4.2],[-4,-1.4],
                            [-4.667,1.4],[1.667,-4.2],[1,1.4],[-3.667,-4.2],
                            [-3.333,4.2],[3.333,-1.4],[2.333,1.4],[-2.667,-1.4],
                            [-3,1.4],[7,-1.4],[4.667,1.4],[-2.333,-4.2],
                            [-2,4.2],[15,-1.4],[10.333,1.4],[-1.333,-1.4]
                            ])[:,0]
        rbg_red = np.array([204,47,107]) # red
        rbg_blue = np.array([0,0,255]) # blue
        td_freq_map = self.td_maps[frame_index]
        Xs = []
        Ys = []
        Zs = []
        Labels = []
        for i in range(td_freq_map.shape[0]):
            longitudes = theta*np.pi / 180
            latitudes = i*0.2* np.pi / 180 
            hypotenuses = td_freq_map[i] * np.cos(longitudes)
            X = hypotenuses * np.sin(latitudes)
            Y = hypotenuses * np.cos(latitudes)
            Z = td_freq_map[i] * np.sin(longitudes)
            Label =  (td_freq_map[i]<self.thred_map[i]).astype('int')   
            Xs.append(X)
            Ys.append(Y)
            Zs.append(Z)
            Labels.append(Label)
        Xs = np.concatenate(Xs)
        Ys = np.concatenate(Ys)
        Zs = np.concatenate(Zs)
        Labels = np.concatenate(Labels)
        color_labels = np.zeros((len(Labels),3))
        color_labels[Labels == 1] = rbg_blue
        color_labels[Labels == 0] = rbg_red
        pcd = op3.geometry.PointCloud()
        pcd.points = op3.utility.Vector3dVector(np.concatenate([Xs.reshape(-1,1),Ys.reshape(-1,1),Zs.reshape(-1,1)],axis = 1))
        pcd.colors = op3.utility.Vector3dVector(color_labels/255)
        return pcd
    
        

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
        self.timing_offset = self.calc_timing_offsets()
        self.omega = self.Data_order[:,0]
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

    def calc_precise_azimuth(self,azimuth):

        # block_number: how many blocks are required to be processed 

        org_azi = azimuth.copy()
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

        precision_azimuth = np.array(precision_azimuth)
        return precision_azimuth # 12 * 32

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

        azimuth_per_block = np.array(azimuth_per_block)
        distances = np.array(distances)*4/1000 # 12*32
        intensities = np.array(intensities) # 12*32

        return distances,intensities, azimuth_per_block # 12*0
 
    
    def gen_td_map(self,frame):
        
        azimuth_ind = np.around(frame[:,0]/0.2).astype('int')
        azimuth_ind[azimuth_ind == 1800] = 0
        theta_ind = frame[:,-1].astype('int')
        td_freq_map = np.zeros((1800,32))
        td_freq_map[azimuth_ind,theta_ind] = frame[:,1]
        return td_freq_map
    
    def cal_angle_diff(self,advance_angle,lagging_angle):
        if advance_angle < lagging_angle:
            return advance_angle + 360 - lagging_angle
        else:
            return advance_angle - lagging_angle
    
    def frame_gen(self):
        initial_azimuth = -1
        cur_azimuth = -1
        culmulative_azimuth = 0
        block_num = 0
        
        while True:
            frame = []
            for ts,buf in self.lidar_reader:
                
                eth = dpkt.ethernet.Ethernet(buf)
                data = eth.data.data.data
                packet_status = eth.data.data.sport
                if packet_status == 2368:
                    if len(data)<1206:
                        continue
                    distances,intensities,azimuth_per_block = self.parse_one_packet(data)
                    
                    azimuth = self.calc_precise_azimuth(azimuth_per_block) # 12*32
                    azimuth += self.Data_order[:,1]
                    if initial_azimuth == -1: #initialization
                        initial_azimuth = azimuth_per_block[0]
                        cur_azimuth = azimuth_per_block[-1]
                        frame.append(np.concatenate([azimuth.flatten().reshape(-1,1),
                                                     distances.flatten().reshape(-1,1),
                                                     intensities.flatten().reshape(-1,1)],axis = 1))
                        block_num += 12
                        culmulative_azimuth += self.cal_angle_diff(azimuth_per_block[-1],azimuth_per_block[0])
                    else:# non-initialization
                        diff = self.cal_angle_diff(azimuth_per_block[-1],cur_azimuth)
                        temp_culmulative_azimuth = culmulative_azimuth + diff 
                        
                        if temp_culmulative_azimuth > 360: 
                            
                            frame_end_index = 0
                            for i in range(len(azimuth_per_block)):
                                diff = self.cal_angle_diff(azimuth_per_block[i],cur_azimuth)                                
                                temp_culmulative_azimuth = culmulative_azimuth + diff
                                if temp_culmulative_azimuth > 360:
                                    frame_end_index = i 
                                    break 
                                else:
                                    block_num += 1
                                    cur_azimuth = azimuth_per_block[i]
                                    culmulative_azimuth = temp_culmulative_azimuth
                                    
                            frame.append(np.concatenate([azimuth[:frame_end_index].flatten().reshape(-1,1),
                                                         distances[:frame_end_index].flatten().reshape(-1,1),
                                                         intensities[:frame_end_index].flatten().reshape(-1,1)],axis = 1))
                            temp = np.concatenate(frame)
                            theta_ind = np.tile(np.arange(32),block_num).reshape(-1,1)
                            temp[:,0][temp[:,0]>360] -= 360
                            temp[:,0][temp[:,0]<0] += 360
                            temp = np.concatenate([temp,theta_ind],axis = 1)
                            frame = []
                            block_num = 12 - frame_end_index  
                            cur_azimuth = azimuth_per_block[-1]
                            diff = self.cal_angle_diff(azimuth_per_block[-1],azimuth_per_block[frame_end_index])                    
                            culmulative_azimuth = diff
                            frame.append(np.concatenate([azimuth[frame_end_index:].flatten().reshape(-1,1),
                                                         distances[frame_end_index:].flatten().reshape(-1,1),
                                                         intensities[frame_end_index:].flatten().reshape(-1,1)],axis = 1))
                            td_freq_map = self.gen_td_map(temp)
                            
                            yield td_freq_map #32*1800
                            
                        else:
                            culmulative_azimuth = temp_culmulative_azimuth
                            frame.append(np.concatenate([azimuth.flatten().reshape(-1,1),
                                                         distances.flatten().reshape(-1,1),
                                                         intensities.flatten().reshape(-1,1)],axis = 1))
                            cur_azimuth = azimuth_per_block[-1]
                            block_num += 12

            
if __name__ == "__main__":
    os.chdir(r'/Users/czhui960/Documents/Lidar/RawLidarData/Vateran')
    collector = RansacCollector(pcap_path=r'./Vateran.pcap',frames_set = np.arange(0,2000,1).astype('int'))
    collector.gen_tdmap()
    collector.gen_thredmap()
    collector.save_tdmap()
    collector.gen_pcdseq(collector.frames_set)