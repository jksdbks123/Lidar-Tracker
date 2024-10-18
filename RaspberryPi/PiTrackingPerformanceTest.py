from Utils import *
from DDBSCAN import Raster_DBSCAN
import cv2
from multiprocessing import Process, Queue, Event, Manager
from LiDARBase import *
from MOT_TD_BCKONLIONE import MOT
import time
import sys

def track_point_clouds(mot,point_cloud_queue,result_queue):
    time_curves = []
    start_tracking_time = time.time()
    frame_count = 0
    while True:
        Td_map =  point_cloud_queue.get()
        # some steps
        time_a = time.time()
        if not mot.if_initialized:
            mot.initialization(Td_map)
            Tracking_pool = mot.Tracking_pool
            Labeling_map = mot.cur_Labeling_map
            time_b = time.time()
        else:
            
            mot.mot_tracking_step(Td_map)
            Tracking_pool = mot.Tracking_pool
            Labeling_map = mot.cur_Labeling_map

            time_b = time.time()
            tracking_time_cost = time_b - time_a
            if (time_b - start_tracking_time) > 120:
                 mot.Off_tracking_pool = {}
                 mot.Tracking_pool = {}
                 mot.Global_id = 0
                 start_tracking_time = time.time()
            time_curves.append(tracking_time_cost * 1000)
            frame_count += 1
            print(f"Tracking time cost: {tracking_time_cost * 1000}",end='\r',flush=True)
            sys.stdout.flush()
        if frame_count > 100:
            # save the time curves
            np.save('time_curves.npy',np.array(time_curves))
            # exit the process
            break
        # result_queue.put((Tracking_pool,Labeling_map,Td_map,time_b - time_a))

def main(pcap_file_path,thred_map_path,win_size,min_samples,eps_dis):
    thred_map = np.load(thred_map_path)
    with Manager() as manger:
        raw_data_queue = manger.Queue(5000) # Packet Queue
        point_cloud_queue = manger.Queue(100)
        tracking_result_queue = manger.Queue() # this is for the tracking results (pt,...)
        tracking_parameter_dict = manger.dict({})
        tracking_parameter_dict['win_size'] = [7,13]
        tracking_parameter_dict['min_samples'] = 5
        tracking_parameter_dict['eps'] = 1.5
        mot = MOT(tracking_parameter_dict, thred_map = thred_map, missing_thred = 10)
        packet_reader_process = Process(target=read_packets_offline, args=(raw_data_queue,pcap_file_path,))
        packet_parser_process = Process(target=parse_packets, args=(raw_data_queue, point_cloud_queue,))
        tracking_prcess = Process(target=track_point_clouds, args=(mot,point_cloud_queue,tracking_result_queue,))
        packet_reader_process.start()
        packet_parser_process.start()
        tracking_prcess.start()

        # terminate all process when the packet reader process is done
        packet_reader_process.join()
        packet_parser_process.join()
        tracking_prcess.join()
        print("All processes are done!")

if __name__ == '__main__':
    pcap_file_path = r"./2024-03-14-23-30-00.pcap"
    thred_map_path = r"./config_files/thred_map.npy"
    win_size = 5
    min_samples = 5
    eps_dis = 0.5
    main(pcap_file_path,thred_map_path,win_size,min_samples,eps_dis)