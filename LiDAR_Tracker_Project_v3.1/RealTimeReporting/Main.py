from multiprocessing import Process, Queue, Event, Manager
import multiprocessing
import socket
import sys
import os
import time
import numpy as np
from multiprocessing import get_context

# Get absolute path of the Interface directory (parent of Utils)
interface_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', r'Interface'))
# Add Interface to sys.path
sys.path.insert(0, interface_path)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add Interface to sys.path
sys.path.insert(0, root_path)
print(sys.path)
from Utils.LiDARBase import * 
from Utils.config import Config
from RaspberryPi.MOT_TD_BCKONLIONE import MOT
from RaspberryPi.Utils import BarDrawer,line_segments_intersect
from RaspberryPi.LiDARBase import parse_packets 

"""
This program is to report volumn counts in real-time trend
"""
def count_traffic_stats(tracking_result_queue,bar_drawer,output_file_dir,data_reporting_interval = 5):
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
    cur_ts = time.time()
    reporting_ts = cur_ts + data_reporting_interval
    print(f'Reporting at {reporting_ts}')
    while True:
        print(f'Get tracking result at {cur_ts}')
        tracking_dic,Labeling_map,Td_map,tracking_cums,ts = tracking_result_queue.get()
        print(f'Get tracking result at {ts}')
        for obj_id in tracking_dic.keys():
            # counting function
            if len(tracking_dic[obj_id].post_seq) > 7:
                prev_pos = tracking_dic[obj_id].post_seq[-6][0].flatten()[:2]
                curr_pos = tracking_dic[obj_id].post_seq[-1][0].flatten()[:2]
                for i in range(len(bar_drawer.line_counts)):
                    if line_segments_intersect(prev_pos, curr_pos, bar_drawer.lines[i][0], bar_drawer.lines[i][1]):
                        cur_time = tracking_dic[obj_id].start_frame + len(tracking_dic[obj_id].mea_seq) - 1
                        # print(f'Line {i} crossed by object {obj_id}, time: {cur_time}, last count time: {self.bar_drawer.last_count_ts[i]}, diff: {cur_time - self.bar_drawer.last_count_ts[i]}')
                        if cur_time - bar_drawer.last_count_ts[i] > 10:
                            bar_drawer.line_counts[i] += 1
                            bar_drawer.last_count_ts[i] = cur_time
                        break
        
        if ts >= reporting_ts:
            print(f'Reporting at {ts}')
            print(f'Line counts: {bar_drawer.line_counts}')
            with open(os.path.join(output_file_dir,cur_ts,'.txt'), 'w') as f:
                for i in range(len(bar_drawer.line_counts)):
                    f.write(f'Line {i}: {bar_drawer.line_counts[i]}\n') 
            reporting_ts += data_reporting_interval * 60

def read_packets_online(port,raw_data_queue):
    sock = socket.socket(socket.AF_INET, # Internet
                    socket.SOCK_DGRAM) # UDP
    sock.bind(('', port))     
    while True:
        print(raw_data_queue.qsize())
        data,addr = sock.recvfrom(1206)
        raw_data_queue.put_nowait((time.time(),data))
        

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    bar_file_path = r'./bar.txt'
    thred_map = np.load(r'./thred_map.npy')
    port = 2380
    mode = "online" 
    data_reporting_interval = 5
    try:
        with multiprocessing.Manager() as manager:  # Ensure Manager starts before processes
            raw_data_queue = manager.Queue()
            point_cloud_queue = manager.Queue()
            tracking_result_queue = manager.Queue()
            tracking_param_update_event = manager.Event()
            tracking_process_stop_event = manager.Event()

            config = Config()
            config.tracking_parameter_dict['win_size'] = [
                config.tracking_parameter_dict['win_width'],
                config.tracking_parameter_dict['win_height']
            ]
            tracking_parameter_dict = manager.dict(config.tracking_parameter_dict)

            bar_drawer = BarDrawer(bar_file_path=bar_file_path)
            mot = MOT(tracking_parameter_dict, thred_map=thred_map, missing_thred=2)

            # Creating processes
            if mode == "online":
                packet_reader_process = Process(target=read_packets_online, args=(port, raw_data_queue,))
            else:  # mode == "offline"
                packet_reader_process = Process(target=read_packets_offline, args=(raw_data_queue, pcap_file_path,))

            packet_parser_process = Process(target=parse_packets, args=(raw_data_queue, point_cloud_queue,))
            tracking_process = Process(target=track_point_clouds, args=(
                tracking_process_stop_event, mot, point_cloud_queue, tracking_result_queue,
                tracking_parameter_dict, tracking_param_update_event
            ))
            traffic_stats_process = Process(target=count_traffic_stats, args=(
                tracking_result_queue, bar_drawer, os.path.join("./", "output_files"), data_reporting_interval
            ))

            # Start processes
            packet_reader_process.start()
            packet_parser_process.start()
            tracking_process.start()
            traffic_stats_process.start()
            print("Processes started!")

            # Wait for termination signal
            while True:
                try:
                    pass  # Keep running
                except KeyboardInterrupt:
                    print("Shutting down processes...")
                    tracking_process_stop_event.set()  # Signal processes to stop cleanly
                    break

            # Cleanup
            for proc in [packet_reader_process, packet_parser_process, tracking_process, traffic_stats_process]:
                proc.terminate()
                proc.join()

            print("Multiprocessing test complete!")

    except Exception as e:
        print("Error:", e)



# def main(thred_map, mode = 'online', port = 2368, pcap_file_path = None, data_reporting_interval = 10, bar_file_path = r'D:\CodeRepos\Lidar-Tracker\RaspberryPi\config_files\bars.txt'):
#     # data reporting interval is in seconds
#     try:
#         with Manager() as manager:
#             # set_start_method("spawn")
#             raw_data_queue = manager.Queue() # Packet Queue
#             point_cloud_queue = manager.Queue()
#             tracking_result_queue = manager.Queue() # this is for the tracking results (pt,...)
#             # traffic_stats_queue = manager.dict({})
#             config = Config()
#             config.tracking_parameter_dict['win_size'] = [config.tracking_parameter_dict['win_width'],config.tracking_parameter_dict['win_height']]
#             tracking_parameter_dict = manager.dict(config.tracking_parameter_dict)
#             tracking_param_update_event = manager.Event()
#             tracking_process_stop_event = manager.Event()
#             bar_drawer = BarDrawer(bar_file_path = bar_file_path)
        
#             mot = MOT(tracking_parameter_dict, thred_map = thred_map, missing_thred = 2)
            
#             # Creating processes for Core 2 and Core 3 tasks
#             if mode == 'online':
#                 packet_reader_process = Process(target=read_packets_online, args=(port,raw_data_queue,))
#             elif mode == 'offline':
#                 packet_reader_process = Process(target=read_packets_offline, args=(raw_data_queue,pcap_file_path,))

#             tracking_prcess = Process(target=track_point_clouds, args=(tracking_process_stop_event,mot,point_cloud_queue,tracking_result_queue,tracking_parameter_dict,tracking_param_update_event,))
#             packet_parser_process = Process(target=parse_packets, args=(raw_data_queue, point_cloud_queue,))
#             packet_reader_process.start()
#             packet_parser_process.start()
#             tracking_prcess.start()
#             traffic_stats_process = Process(target=count_traffic_stats, args=(tracking_result_queue,bar_drawer,os.path.join('./','output_files'),data_reporting_interval,))
#             traffic_stats_process.start()
            
            
#             # # Cleanup
#             # packet_reader_process.terminate()
#             # packet_parser_process.terminate()
#             # tracking_prcess.terminate()
#             # traffic_stats_process.terminate()
            
#             # packet_reader_process.join()
#             # packet_parser_process.join()
#             # tracking_prcess.join()
#             # traffic_stats_process.join()

#     except KeyboardInterrupt :
#         packet_reader_process.terminate()
#         packet_parser_process.terminate()
#         tracking_prcess.terminate()
#         traffic_stats_process.terminate()
#         packet_reader_process.join()
#         packet_parser_process.join()
#         tracking_prcess.join()
#         traffic_stats_process.join()

# if __name__ == '__main__':
#     # set_start_method("spawn")
#     multiprocessing.set_start_method("spawn") 
#     thred_map = np.load(r'./thred_map.npy')
#     main(thred_map = thred_map, mode = 'online', bar_file_path = r'./bar.txt')