
# import sys
# import os

# # Get the path to the upper-level directory

# dependency_folder = os.path.abspath(os.path.join(os.path.dirname(__file__),'Utils'))
# # Add the path to sys.path
# if dependency_folder not in sys.path:
#     sys.path.append(dependency_folder)

from Utils.LiDARBase import *
from datetime import datetime

def get_pcap_start_time(pcap_file):
    eth_reader = load_pcap(pcap_file)
    while True:
        try:
            ts,buf = next(eth_reader)
            eth = dpkt.ethernet.Ethernet(buf)
        except StopIteration:
            return None
            
        if eth.type == 2048: # for ipv4
            if (type(eth.data.data) == dpkt.udp.UDP):# for ipv4
                data = eth.data.data.data
                packet_status = eth.data.data.sport
                if packet_status == 2368:
                    if len(data) != 1206:
                        continue
                    # convert from unix timestamp to datetime
                    ts = datetime.fromtimestamp(ts)
                    return ts
                        