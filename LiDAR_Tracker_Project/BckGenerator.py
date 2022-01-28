import os
import numpy as np
import argparse
from BfTableGenerator import RansacCollector


parser = argparse.ArgumentParser(description='This is a program to generate bckfiles')
parser.add_argument('-i','--input', help='path that contains .pcap file', required=True)
parser.add_argument('-o','--output', help='specified output path', required=True)
args = parser.parse_args()


bck_path = args.input
output_path = args.output
pcaps_list = os.listdir(bck_path)
pcaps_list = [p for p in pcaps_list if 'pcap' in p.split('.')]
aggregated_maps_multiple = []
for i in range(len(pcaps_list)):
    collector = RansacCollector(pcap_path=os.path.join(bck_path,pcaps_list[i]),output_file_path=bck_path,update_frame_num=2000)
    collector.gen_tdmap()
    aggregated_maps_multiple.append(collector.aggregated_map[np.random.choice(np.arange(0,2000),size = int(6000/len(pcaps_list)) ,replace = False)])
collector.aggregated_map = np.concatenate(aggregated_maps_multiple,axis = 0)
collector.gen_thredmap(d = 0.5 ,thred_s = 0.54,N = 10,delta_thred = 1e-3,step = 0.1)
threshold_map = collector.thred_map

np.save(os.path.join(output_path,'bck_map.npy'),threshold_map)