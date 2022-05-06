import argparse
from MOT_TD_BCKONLIONE import MOT
from Utils import *
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='This is a program to visulize the 3D tracking')
    parser.add_argument('-i','--input', help='path to the .pcap file', required=True)
    parser.add_argument('-c','--cali', help='path to the .calibration file', required=True)
    args = parser.parse_args()
    config_path = args.cali 
    with open(config_path) as f:
        params = json.load(f)
    mot = MOT(args.input,r'./',**params,if_vis=True)
    mot.initialization()
    mot.mot_tracking()