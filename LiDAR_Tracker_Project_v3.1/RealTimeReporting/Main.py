
import multiprocessing

from multiprocessing import Process
import socket
import sys
import os
import time
import numpy as np

# Get absolute path of the Interface directory (parent of Utils)
interface_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', r'Interface'))
# Add Interface to sys.path
sys.path.insert(0, interface_path)
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# Add Interface to sys.path
sys.path.insert(0, root_path)
# rasp_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', r'RaspberryPi'))
# Add Interface to sys.path
# sys.path.insert(0, rasp_path)
# print(sys.path)
from Utils.LiDARBase import *
# from RaspberryPi.LiDARBase import * 
from Utils.config import Config
from RaspberryPi.LiDARBase import parse_packets,track_point_clouds,read_packets_online
from RaspberryPi.MOT_TD_BCKONLIONE import MOT
from RaspberryPi.Utils import BarDrawer,line_segments_intersect
from RaspberryPi.GenBckFile import gen_bckmap



import subprocess

def free_udp_port(port):
    """Find and kill any process using a given UDP port."""
    try:
        # Find the process using the port
        result = subprocess.run(
            ["sudo", "netstat", "-tulnp"], capture_output=True, text=True
        )

        for line in result.stdout.split("\n"):
            if f":{port}" in line and "udp" in line:
                parts = line.split()
                pid = parts[-1].split("/")[0]  # Extract the PID
                print(f"🔹 Port {port} is in use by PID {pid}, terminating...")

                # Kill the process using the port
                subprocess.run(["sudo", "kill", "-9", pid])
                print(f"✅ Successfully freed port {port}")

                return True

        print(f"✅ Port {port} is already free")
        return False

    except Exception as e:
        print(f"❌ Error freeing port {port}: {e}")
        return False


"""
This program is to report volumn counts in real-time trend
"""
def count_traffic_stats(tracking_result_queue,bar_drawer,output_file_dir,data_reporting_interval = 5):
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
    cur_ts = time.time()
    reporting_ts = cur_ts + data_reporting_interval * 60
    print(f'Reporting at {reporting_ts}')
    print(len(bar_drawer.line_counts))
    while True:
        # print(f'Get tracking result at {cur_ts}')
        tracking_dic,Labeling_map,Td_map,tracking_cums,ts,bf_time, clustering_time, association_time = tracking_result_queue.get()
        # constant show the realtime tracking_cums
        sys.stdout.write(f'\rData Processing Speed (sec): {clustering_time:.3f}, {bf_time:.3f}, {association_time:.3f}')
        sys.stdout.flush()
        # print(f'Get tracking result at {ts}')
        for obj_id in tracking_dic.keys():
            # counting function
            if len(tracking_dic[obj_id].post_seq) > 7:
                prev_pos = tracking_dic[obj_id].post_seq[-6][0].flatten()[:2]
                curr_pos = tracking_dic[obj_id].post_seq[-1][0].flatten()[:2]
                for i in range(len(bar_drawer.line_counts)):
                    if line_segments_intersect(prev_pos, curr_pos, bar_drawer.lines[i][0], bar_drawer.lines[i][1]):
                        cur_time = tracking_dic[obj_id].start_frame + len(tracking_dic[obj_id].mea_seq) - 1
                        if cur_time - bar_drawer.last_count_ts[i] > 10:
                            bar_drawer.line_counts[i] += 1
                            bar_drawer.last_count_ts[i] = cur_time
                        break
        
        if ts >= reporting_ts:
            print(f'Reporting at {ts}')
            # print(f'Line counts: {bar_drawer.line_counts}')
            # convert cur_ts to datetime string
            cur_ts_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(cur_ts))
            output_path = os.path.join(output_file_dir,cur_ts_str + '.txt')
            
            with open(output_path, 'w') as f:
                for i in range(len(bar_drawer.line_counts)):
                    print(f'Line {i}: {bar_drawer.line_counts[i]}')
                    f.write(f'Line {i}: {bar_drawer.line_counts[i]}\n') 
                    bar_drawer.line_counts[i] = 0
            reporting_ts += data_reporting_interval * 60
            cur_ts = ts

def read_packets_online(port,raw_data_queue):
    sock = socket.socket(socket.AF_INET, # Internet
                    socket.SOCK_DGRAM) # UDP
    sock.bind(('', port))     
    while True:
        data,addr = sock.recvfrom(1206)
        raw_data_queue.put_nowait((time.time(),data))
        
def clear_queue(queue):
    """Clears all items in a multiprocessing queue."""
    while not queue.empty():
        try:
            queue.get_nowait()
        except Exception:
            break  # In case of race conditions

def background_update_process(thred_map_dict, background_point_copy_event, background_point_cloud_queue, update_interval):
    """Periodically generates a new background map and updates the tracking process."""
    while True:
        background_point_copy_event.set()  # Copy point cloud data to background queue
        time.sleep(update_interval)  # Wait for update interval (e.g., 10 minutes)
        background_point_copy_event.clear()  # Stop copying point cloud data
        print("Starting background update process...")
        aggregated_maps = []
        while not background_point_cloud_queue.empty():
            try:
                aggregated_maps.append(background_point_cloud_queue.get_nowait())
            except Exception:
                break

        if aggregated_maps:
            aggregated_maps = np.array(aggregated_maps)
            new_thred_map = gen_bckmap(aggregated_maps, N=10, d_thred=0.1, bck_n=3)
            print("Generated new background map!")

            # Update the shared thred_map safely
            thred_map_dict["thred_map"] = new_thred_map
            print("Updated thred_map in tracking process.")

def run_processes(manager, raw_data_queue, point_cloud_queue, background_point_cloud_queue, tracking_result_queue, port, bar_file_path, data_reporting_interval, background_data_generating_time = 60, background_update_interval = 60):
    """
    Runs the processes including real-time tracking and periodic background updating.
    background_data_generating_time (sec): Time in seconds to generate a background map.
    background_update_interval (sec): Time in seconds to update the background map.
    
    """
    try:
        # Step 1: **Initial Background Data Generation**
        free_udp_port(port)
        print("Starting initial background data collection...")

        packet_reader_process = multiprocessing.Process(target=read_packets_online, args=(port, raw_data_queue,))
        packet_parser_process = multiprocessing.Process(target=parse_packets, args=(raw_data_queue, background_point_cloud_queue,))

        packet_reader_process.start()
        packet_parser_process.start()

        time.sleep(background_data_generating_time)  # Initial background generation time

        # Terminate background processes
        packet_reader_process.terminate()
        packet_parser_process.terminate()
        packet_reader_process.join()
        packet_parser_process.join()

        # Process collected point cloud data for initial background
        aggregated_maps = []
        while not background_point_cloud_queue.empty():
            try:
                aggregated_maps.append(background_point_cloud_queue.get_nowait())
            except Exception:
                break

        aggregated_maps = np.array(aggregated_maps)
        print("Generating initial background map...")
        initial_thred_map = gen_bckmap(aggregated_maps, N=10, d_thred=0.1, bck_n=3)

        # Clear queues instead of redefining them
        clear_queue(raw_data_queue)
        clear_queue(background_point_cloud_queue)
        free_udp_port(port)

        print("Starting real-time monitoring...")

        # Step 2: **Real-Time Tracking**
        tracking_param_update_event = manager.Event()
        tracking_process_stop_event = manager.Event()
        background_update_event = manager.Event()
        background_point_copy_event = manager.Event() # copy point cloud from end of parsing process to background_point_cloud_queue

        config = Config()
        config.tracking_parameter_dict['win_size'] = [
            config.tracking_parameter_dict['win_width'],
            config.tracking_parameter_dict['win_height']
        ]
        tracking_parameter_dict = manager.dict(config.tracking_parameter_dict)

        bar_drawer = BarDrawer(bar_file_path=bar_file_path)

        # Shared dictionary for thred_map updates
        thred_map_dict = manager.dict({"thred_map": initial_thred_map})

        mot = MOT(tracking_parameter_dict, thred_map_dict["thred_map"], missing_thred=2)

        # Creating processes
        packet_reader_process = multiprocessing.Process(target=read_packets_online, args=(port, raw_data_queue,))
        packet_parser_process = multiprocessing.Process(target=parse_packets, args=(raw_data_queue, point_cloud_queue,background_point_cloud_queue,background_point_copy_event))
        tracking_process = multiprocessing.Process(target=track_point_clouds, args=(
            tracking_process_stop_event, mot, point_cloud_queue, tracking_result_queue,
            tracking_parameter_dict, tracking_param_update_event,background_update_event
        ))
        traffic_stats_process = multiprocessing.Process(target=count_traffic_stats, args=(
            tracking_result_queue, bar_drawer, os.path.join("./", "output_files"), data_reporting_interval
        ))

        background_update_proc = multiprocessing.Process(target=background_update_process, args=(
            thred_map_dict,background_point_copy_event ,background_point_cloud_queue, background_update_interval  # Update every 10 minutes
        ))

        # Start processes
        packet_reader_process.start()
        packet_parser_process.start()
        tracking_process.start()
        traffic_stats_process.start()
        background_update_proc.start()
        
        print("Processes started!")

        # Wait for termination signal
        while True:
            try:
                pass  # Keep running
            except KeyboardInterrupt:
                print("Shutting down processes...")
                tracking_process_stop_event.set()  # Signal processes to stop cleanly
                # Cleanup
                for proc in [packet_reader_process, packet_parser_process, tracking_process, traffic_stats_process, background_update_proc]:
                    proc.terminate()
                    proc.join()
                print("Multiprocessing test complete!")
                break

    except KeyboardInterrupt as e:
        print(e)
        print("Shutting down processes...")
        tracking_process_stop_event.set()  # Signal processes to stop cleanly
        # Cleanup
        for proc in [packet_reader_process, packet_parser_process, tracking_process, traffic_stats_process, background_update_proc]:
            proc.terminate()
            proc.join()
        print("Multiprocessing test complete!")

if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")
    bar_file_path = r'./bars.txt'
    port = 2380
    mode = "online" 
    data_reporting_interval = 1
    background_data_generting_time = 60 # sec
    
    with multiprocessing.Manager() as manager:
        # Define queues **once** and reuse them
        raw_data_queue = manager.Queue()
        point_cloud_queue = manager.Queue()
        tracking_result_queue = manager.Queue()
        background_point_cloud_queue = manager.Queue()
        run_processes(manager, raw_data_queue, point_cloud_queue, background_point_cloud_queue, tracking_result_queue, port, bar_file_path, data_reporting_interval, background_data_generting_time)
