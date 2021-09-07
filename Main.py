import argparse
from MOT import MOT
from Utils import *
import json
parser = argparse.ArgumentParser(description='This is a program generating trajectories from .pcap files')
parser.add_argument('-p','--pcap', help='.pcap file path', required=True)
parser.add_argument('-o','--output', help='designed output path', required=True)
parser.add_argument('-c','--config', help='designed config path', required=True)
args = parser.parse_args()


pcap_path = args.pcap
output_file_path = args.output
with open(args.config) as f:
    params = json.load(f)

mot = MOT(pcap_path,output_file_path,**params)
mot.initialization()
mot.mot_tracking(A,P,H,Q,R)
mot.save_result()
