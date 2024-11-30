
from LiDARBase import *
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
                        