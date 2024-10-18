import numpy as np
import dpkt

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

timing_offset = calc_timing_offsets()
omega = Data_order[:,0]
theta = np.sort(omega)
azimuths = np.arange(0,360,0.2)
arg_omega = np.argsort(omega)

def read_firing_data(data):
    block_id = data[0] + data[1]*256
    azimuth = (data[2] + data[3] * 256) / 100 # degree
    firings = data[4:].reshape(32, 3) 
    distances = firings[:, 0] + firings[:, 1] * 256 # mm 
    intensities = firings[:, 2] # 0-255
    return distances, intensities, azimuth #(1,0)

def read_uint32(data, idx):
    return data[idx] + data[idx+1]*256 + data[idx+2]*256*256 + data[idx+3]*256*256*256

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


class TDmapLoader():
    def __init__(self,pcap_file_path): 
        
        self.file_path = pcap_file_path
        self.packets_gen = read_packets_offline(pcap_file_path)
        packets_gen = read_packets_offline(pcap_file_path)
        packet = next(packets_gen)
        Initial_ts,raw_packet = packet
        self.Init_ts = Initial_ts
        packets_gen = read_packets_offline(pcap_file_path)
        self.td_gen = parse_packets(packets_gen)