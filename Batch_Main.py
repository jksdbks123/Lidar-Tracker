import argparse
from MOT_TR import MOT
from Utils import *
import pandas as pd
import json
import os



parser = argparse.ArgumentParser(description='This is a program generating trajectories from .pcap files')
parser.add_argument('-i','--input', help='path that contains .pcap file', required=True)
parser.add_argument('-o','--output', help='specified output path', required=True)
args = parser.parse_args()

input_path = args.input
dir_lis = os.listdir(input_path)
output_file_path = args.output
pcap_paths = []
output_file_paths = []

for f in dir_lis:
    if 'pcap' in f.split('.'):
        pcap_path = os.path.join(input_path,f)
        output_path = os.path.join(output_file_path,f.split('.')[0])
        pcap_paths.append(pcap_path)
        output_file_paths.append(output_path)

if len(pcap_paths) == 0:
    print('Pcap file is not detected')

config_path = os.path.join(input_path,'config.json')
ref_LLH_path,ref_xyz_path = os.path.join(input_path,'LLE_ref.csv'),os.path.join(input_path,'xyz_ref.csv')
ref_LLH,ref_xyz = np.array(pd.read_csv(ref_LLH_path)),np.array(pd.read_csv(ref_xyz_path))
ref_LLH[:,[0,1]] = ref_LLH[:,[0,1]] * np.pi/180
ref_LLH[:,2] = ref_LLH[:,2]/3.2808

with open(config_path) as f:
    params = json.load(f)

for i,p in enumerate(pcap_paths):
    
    if not os.path.exists(output_file_paths[i]):
        os.makedirs(output_file_paths[i])
    mot = MOT(p,output_file_paths[i],**params)
    mot.initialization()
    mot.mot_tracking(A,P,H,Q,R)
    mot.save_result(ref_LLH,ref_xyz)
