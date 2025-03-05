from multiprocessing import Process, Queue, Event, Manager
from multiprocessing import set_start_method
import socket
import sys
import os

# Get absolute path of the Interface directory (parent of Utils)
interface_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', r'Interface'))
# Add Interface to sys.path
sys.path.insert(0, interface_path)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add Interface to sys.path
sys.path.insert(0, root_path)
print(sys.path)
from Utils.LiDARBase import * 
from RaspberryPi.MOT_TD_BCKONLIONE import MOT
from RaspberryPi.Utils import BarDrawer

"""
This program is to report volumn counts in real-time trend
"""


def main(thred_map, mode = 'online', port = 2368, pcap_file_path = None):
    # pcap_file_path = r'/Users/zhihiuchen/Documents/Data/2019-12-21-7-30-0.pcap'
    try:
        with Manager() as manger:
            # set_start_method('fork',force=True)
            raw_data_queue = manger.Queue() # Packet Queue
            point_cloud_queue = manger.Queue()
            tracking_result_queue = manger.Queue() # this is for the tracking results (pt,...)
            tracking_parameter_dict = manger.dict({})
            tracking_param_update_event = Event()
            tracking_process_stop_event = Event()
            
            mot = MOT(tracking_parameter_dict, thred_map = thred_map, missing_thred = 2)
            
            # Creating processes for Core 2 and Core 3 tasks
            if mode == 'online':
                sock = socket.socket(socket.AF_INET, # Internet
                                socket.SOCK_DGRAM) # UDP
                sock.bind(('', port)) 
                packet_reader_process = Process(target=read_packets_online, args=(sock,raw_data_queue,))
            elif mode == 'offline':
                packet_reader_process = Process(target=read_packets_offline, args=(raw_data_queue,pcap_file_path,))

            packet_parser_process = Process(target=parse_packets, args=(raw_data_queue, point_cloud_queue,))
            packet_reader_process.start()
            packet_parser_process.start()
            tracking_prcess = Process(target=track_point_clouds, args=(tracking_process_stop_event,mot,point_cloud_queue,tracking_result_queue,tracking_parameter_dict,tracking_param_update_event,))
            tracking_prcess.start()
            
            
            # Cleanup
            packet_reader_process.terminate()
            packet_parser_process.terminate()
            
            packet_reader_process.join()
            packet_parser_process.join()

    except KeyboardInterrupt :
        packet_reader_process.terminate()
        packet_parser_process.terminate()
        packet_reader_process.join()
        packet_parser_process.join()

if __name__ == '__main__':
    print('Starting the main function')