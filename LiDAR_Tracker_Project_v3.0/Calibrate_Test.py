import argparse
from MOT_TD_BCKONLIONE import MOT
from Utils import *
import pandas as pd
import json
import os

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='This is a program generating trajectories from .pcap files')
    parser.add_argument('-i','--input', help='path that contains .pcap file', required=True)
    parser.add_argument('-p','--pcap_name', help='.pcap name', required=True)
    args = parser.parse_args()
    input_path = args.input
    pcap_name = args.pcap_name
    calibration_path = os.path.join(input_path,'Calibration')
    pcap_path = os.path.join(input_path,pcap_name)
    config_path = os.path.join(calibration_path,'config.json')

    with open(config_path) as f:
        params = json.load(f)

    mot = MOT(pcap_path,'./',**params,if_vis=True)
    mot.initialization()
    mot.mot_tracking()