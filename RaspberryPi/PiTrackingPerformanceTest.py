from Utils import *
from DDBSCAN import Raster_DBSCAN
import cv2

def track_point_clouds(stop_event,mot,point_cloud_queue,result_queue,tracking_parameter_dict,tracking_param_update_event):
    start_tracking_time = time.time()
    while not stop_event.is_set():
        Td_map =  point_cloud_queue.get()
        # some steps
        time_a = time.time()
        if not mot.if_initialized:
            mot.initialization(Td_map)
            Tracking_pool = mot.Tracking_pool
            Labeling_map = mot.cur_Labeling_map
            time_b = time.time()
        else:
            if tracking_param_update_event.is_set():
                mot.db = Raster_DBSCAN(window_size=tracking_parameter_dict['win_size'],eps = tracking_parameter_dict['eps'], min_samples= tracking_parameter_dict['min_samples'],Td_map_szie=(32,1800))
                tracking_param_update_event.clear()
            
            mot.mot_tracking_step(Td_map)
            Tracking_pool = mot.Tracking_pool
            Labeling_map = mot.cur_Labeling_map

            time_b = time.time()
            if (time_b - start_tracking_time) > 120:
                 mot.Off_tracking_pool = {}
                 mot.Tracking_pool = {}
                 mot.Global_id = 0
                 start_tracking_time = time.time()
            

        result_queue.put((Tracking_pool,Labeling_map,Td_map,time_b - time_a))