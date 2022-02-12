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

config_path = os.path.join(calibration_path,'config.json')
ref_LLH_path,ref_xyz_path = os.path.join(calibration_path,'LLE_ref.csv'),os.path.join(calibration_path,'xyz_ref.csv')
ref_LLH,ref_xyz = np.array(pd.read_csv(ref_LLH_path)),np.array(pd.read_csv(ref_xyz_path))
if len(np.unique(ref_xyz[:,2])) == 1:
    np.random.seed(1)
    offset = np.random.normal(-0.521,3.28,len(ref_LLH))
    ref_xyz[:,2] += offset
    ref_LLH[:,2] += offset
ref_LLH[:,[0,1]] = ref_LLH[:,[0,1]] * np.pi/180
ref_LLH[:,2] = ref_LLH[:,2]/3.2808
bck_map_path = os.path.join(calibration_path,'bck_map.npy')
bck_map = np.load(bck_map_path)
plane_model_path = os.path.join(calibration_path,'plane_model.npy')
plane_model = np.load(plane_model_path)
with open(config_path) as f:
    params = json.load(f)

for i,p in enumerate(pcap_paths):
    print(p)
    if not os.path.exists(output_file_paths[i]):
        os.makedirs(output_file_paths[i])
    mot = MOT(p,output_file_paths[i],**params)
    mot.initialization(bck_map)
    mot.mot_tracking(A,plane_model)
    mot.save_result(ref_LLH,ref_xyz)


