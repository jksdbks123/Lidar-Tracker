import dpkt
import numpy as np

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

def load_pcap(file_path):
    try:
        fpcap = open(file_path, 'rb')
        eth_reader = dpkt.pcap.Reader(fpcap)
    except Exception as ex:
        print(str(ex))
    return eth_reader

def parse_one_packet(data):
    data = np.frombuffer(data, dtype=np.uint8).astype(np.uint32)
    blocks = data[0:1200].reshape(12, 100)
    Timestamp = read_uint32(data[1200:1204],0)
    distances = []#12*32
    intensities = []#12*32
    azimuth_per_block = [] #(12,0)
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

def get_pcd_uncolored(Td_map):

    Xs = []
    Ys = []
    Zs = []
    for i in range(Td_map.shape[0]):
        longitudes = theta[i]*np.pi / 180
        latitudes = azimuths * np.pi / 180 
        hypotenuses = Td_map[i] * np.cos(longitudes)
        X = hypotenuses * np.sin(latitudes)
        Y = hypotenuses * np.cos(latitudes)
        Z = Td_map[i] * np.sin(longitudes)
        Xs.append(X)
        Ys.append(Y)
        Zs.append(Z)

    Xs = np.concatenate(Xs)
    Ys = np.concatenate(Ys)
    Zs = np.concatenate(Zs)
    
    XYZ = np.c_[Xs,Ys]
    XYZ = XYZ[(XYZ[:,0] != 0)&(XYZ[:,1] != 0)]
    return XYZ