import argparse
from MOT_TR_FIXBCK_NEAREST import MOT
from Utils import *
import pandas as pd
import json
import os




parser = argparse.ArgumentParser(description='This is a program generating trajectories from .pcap files')
parser.add_argument('-i','--input', help='path that contains .pcap file', required=True)
parser.add_argument('-o','--output', help='specified output path', required=True)
args = parser.parse_args()

input_path = args.input
calibration_path = os.path.join(input_path,'Calibration')
dir_lis = os.listdir(input_path)
pcap_path = 'None'
for f in dir_lis:
    if 'pcap' in f.split('.'):
        pcap_path = os.path.join(input_path,f)
if pcap_path == 'None':
    print('Pcap file is not detected')
output_file_path = args.output
config_path = os.path.join(calibration_path,'config.json')
ref_LLH_path,ref_xyz_path = os.path.join(calibration_path,'LLE_ref.csv'),os.path.join(calibration_path,'xyz_ref.csv')
ref_LLH,ref_xyz = np.array(pd.read_csv(ref_LLH_path)),np.array(pd.read_csv(ref_xyz_path))

ref_LLH[:,2] += np.arange(1,len(ref_LLH)+1)*0.1
if len(np.unique(ref_xyz[:,2])) == 1:
    offset = np.random.normal(-0.521,3.28,len(ref_LLH))
    ref_xyz[:,2] += offset

ref_LLH[:,[0,1]] = ref_LLH[:,[0,1]] * np.pi/180
ref_LLH[:,2] = ref_LLH[:,2]/3.2808

bck_map_path = os.path.join(calibration_path,'bck_map.npy')
bck_map = np.load(bck_map_path)

with open(config_path) as f:
    params = json.load(f)

print(params)
mot = MOT(pcap_path,output_file_path,**params)
mot.initialization(bck_map)
mot.mot_tracking(A)
mot.save_result(ref_LLH,ref_xyz)
#test