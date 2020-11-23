from tqdm import tqdm_notebook
import datetime
import dpkt
import numpy as np
import matplotlib.pyplot as plt
import os

class LidarLoader():
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
        org_azi = azimuth.copy()

        precision_azimuth = []
        # iterate through each block
        for n in range(12): # n=0..11
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
        return precision_azimuth

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
        Timestamp = self.read_uint32(data[1200:1204],0)
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
            ## Note: all these arrray have th same size, number of firing in one packet
        azimuth = self.calc_precise_azimuth(azimuth_per_block).reshape(12, 32) # 12*32
        #offset
        azimuth += self.Data_order[:,1]
        distances = np.array(distances)*4/1000 # 12*32
        intensities = np.array(intensities) # 12*32

        # now calculate the cartesian coordinate of each point
        X, Y, Z = self.calc_cart_coord(distances, azimuth)

        # calculating timestamp [microsec] of each firing
        timestamps = Timestamp + self.timing_offset

        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()
        intensities = intensities.flatten()
        azimuth = azimuth.flatten()
        timestamps = timestamps.flatten()
        distances = distances.flatten()

        # remeber the last azimuth for roll over checking
        return X, Y, Z, intensities, azimuth, timestamps, distances

    def frame_gen(self):
        while True:
            cur_culmulative_fires = 0
            frame = []
            flag = True
            for ts,buf in self.lidar_reader:
                eth = dpkt.ethernet.Ethernet(buf)
                data = eth.data.data.data
                packet_status = eth.data.data.sport
                if packet_status == 2368:
                    cur_culmulative_fires+=1
                    if len(data)<1206:
                        continue
                    X,Y,Z,I,A,T,D = self.parse_one_packet(data)
                    A[A>360] -= 360
                    frame.append(np.concatenate([T.reshape((-1,1)),
                                                    X.reshape((-1,1)),
                                                    Y.reshape((-1,1)),
                                                    Z.reshape((-1,1)),
                                                    D.reshape((-1,1)),
                                                    I.reshape((-1,1))],axis = 1))
                    if cur_culmulative_fires==150:
                        cur_culmulative_fires = 0
                        temp = np.concatenate(frame)
                        temp = temp[temp[:,4]!=0] # filter out sky laser
                        frame = []
                        yield temp    
                else:
                    continue
if __name__ == "__main__":
    os.chdir(r'/Users/czhui960/Documents/Lidar/to ZHIHUI/US 395')
    file_path  = os.listdir()[-4]
    lidar_reader = LidarLoader(file_path)
    frame_gen = lidar_reader.frame_gen()
    print(next(frame_gen).shape)
    print(next(frame_gen).shape)
