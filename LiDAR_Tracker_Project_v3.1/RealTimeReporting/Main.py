
import multiprocessing
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
from queue import Empty, Full  # Standard exceptions


def safe_queue_get(q, timeout=5, default=None, queue_name="queue"):
    """
    Safely get an item from the queue with timeout.
    Returns `default` if queue is empty.
    """
    try:
        item = q.get(timeout=timeout)
        return item
    except Empty:
        print(f"[WARNING] {queue_name}: get() timed out after {timeout}s — queue is empty.")
        return default
    except Exception as e:
        print(f"[ERROR] {queue_name}: unexpected exception during get(): {e}")
        return default


def safe_queue_put(q, item, timeout=5, queue_name="queue"):
    """
    Safely put an item into the queue with timeout.
    Returns True if success, False if failed (queue full or error).
    """
    try:
        q.put(item, timeout=timeout)
        return True
    except Full:
        print(f"[WARNING] {queue_name}: put() timed out after {timeout}s — queue is full.")
        return False
    except Exception as e:
        print(f"[ERROR] {queue_name}: unexpected exception during put(): {e}")
        return False
    
def clear_queue(queue):
    """Clears all items in a multiprocessing queue."""
    while not queue.empty():
        try:
            queue.get_nowait()
        except Exception:
            break  # In case of race conditions

def process_health_monitor(process_list, check_interval=20):
    """Monitor the health of all processes using their PIDs."""
    
    # Ensure we only track processes with valid PIDs
    pid_list = {proc.pid: proc.name for proc in process_list if proc.pid is not None}
    print(f"[INFO] Monitoring {len(pid_list)} processes: {pid_list}")
    if not pid_list:
        print("[ERROR] No valid process PIDs found! Health monitor exiting.")
        return  # Exit if no processes to monitor

    while True:
        time.sleep(check_interval)

        for pid, name in pid_list.items():
            if pid is None:
                print(f"[WARNING] Process {name} has no valid PID. Skipping.")
                continue

            try:
                # Send signal 0 (checks if process exists)
                os.kill(pid, 0)
            except ProcessLookupError:
                print(f"[ERROR] Process {name} (PID {pid}) has died unexpectedly!")
            except Exception as e:
                print(f"[ERROR] Unexpected issue checking {name} (PID {pid}): {e}")

        # print("[INFO] All processes checked. Running normally.")

def queue_monitor_process(raw_data_queue, point_cloud_queue, tracking_result_queue, max_size=100, check_interval=5):
    """Monitors queue sizes to detect overflow issues."""
    while True:
        raw_size = raw_data_queue.qsize()
        pc_size = point_cloud_queue.qsize()
        track_size = tracking_result_queue.qsize()

        print(f"[Queue Monitor] Raw Data: {raw_size}, Point Cloud: {pc_size}, Tracking: {track_size}")

        # Detect potential overflow
        if raw_size > max_size:
            print(f"[WARNING] Raw data queue size {raw_size} exceeds {max_size}, clearing...")
            clear_queue(raw_data_queue)

        if pc_size > max_size:
            print(f"[WARNING] Point cloud queue size {pc_size} exceeds {max_size}, clearing...")
            clear_queue(point_cloud_queue)

        if track_size > max_size:
            print(f"[WARNING] Tracking result queue size {track_size} exceeds {max_size}, clearing...")
            clear_queue(tracking_result_queue)

        time.sleep(check_interval)

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

def report_results(tracking_result_queue,output_file_dir):
    """
    This function is used to constantly report the tracking results and save them to a file.
    Logic: as long as tracking_result_queue is not empty, keep reporting and saving the results.
    """
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
    cur_ts = time.time()
    reporting_ts = cur_ts + data_reporting_interval * 60
    print(f'Reporting at {reporting_ts}')
    # print(len(bar_drawer.line_counts))
    while True:
        cur_ts_str = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(cur_ts))
        output_path = os.path.join(output_file_dir,cur_ts_str + '.txt')
        
        with open(output_path, 'w') as f:
            for i in range(len(bar_drawer.line_counts)):
                print(f'Line {i}: {bar_drawer.line_counts[i]}')
                f.write(f'Line {i}: {bar_drawer.line_counts[i]}\n') 
                bar_drawer.line_counts[i] = 0
    pass

"""
This program is to report volumn counts in real-time trend
"""
def count_traffic_stats(tracking_result_queue,tracking_pool_dict,bar_drawer,data_update_interval = 5):
    cur_ts = time.time()
    update_ts = cur_ts + data_update_interval * 60

    while True:
        # print(tracking_pool_dict)
        for obj_id in tracking_pool_dict.keys():
            # counting function
            if len(tracking_pool_dict[obj_id].post_seq) > 4:
                prev_pos = tracking_pool_dict[obj_id].post_seq[-3][0].flatten()[:2]
                curr_pos = tracking_pool_dict[obj_id].post_seq[-1][0].flatten()[:2]
                for i in range(len(bar_drawer.line_counts)):
                    if line_segments_intersect(prev_pos, curr_pos, bar_drawer.lines[i][0], bar_drawer.lines[i][1]):
                        cur_time = tracking_pool_dict[obj_id].start_frame + len(tracking_pool_dict[obj_id].mea_seq) - 1
                        if cur_time - bar_drawer.last_count_ts[i] > 5:
                            bar_drawer.line_counts[i] += 1
                            bar_drawer.last_count_ts[i] = cur_time
                        break
        # report_dict = {'counting_res':{},'time':cur_ts}
        # for i in range(len(bar_drawer.line_counts)):
        #     report_dict[f'Bar {i}'] = bar_drawer.line_counts[i]
        # safe_queue_put(tracking_result_queue, report_dict, queue_name="tracking_result_queue")
        if cur_ts >= update_ts:
            print(f'Update at {cur_ts}')
            for i in range(len(bar_drawer.line_counts)):
                print(f'Line {i}: {bar_drawer.line_counts[i]}')
                bar_drawer.line_counts[i] = 0
            cur_ts = time.time()
            update_ts = cur_ts + data_update_interval * 60
    

def background_update_process(thred_map_dict, background_point_copy_event, background_point_cloud_queue, background_update_interval, background_data_generating_time,background_update_event):
    """Periodically generates a new background map and updates the tracking process."""
    # print("Background update process started...")
    while True:
        time.sleep(background_update_interval)
        background_point_copy_event.set()  # Copy point cloud data to background queue
        # print(background_point_copy_event.is_set(),'a')
        time.sleep(background_data_generating_time)  # Wait for update interval (e.g., 10 minutes)
        background_point_copy_event.clear()  # Stop copying point cloud data
        aggregated_maps = []
        while not background_point_cloud_queue.empty():
            Td_map = safe_queue_get(background_point_cloud_queue, default=None)
            aggregated_maps.append(Td_map)
            
        # print('Frames to generate background:',len(aggregated_maps))
        if len(aggregated_maps) > 0:
            aggregated_maps = np.array(aggregated_maps)
            new_thred_map = gen_bckmap(aggregated_maps, N=10, d_thred=0.1, bck_n=3)
            # print("Generated new background map!")
            # Update the shared thred_map safely
            thred_map_dict["thred_map"] = new_thred_map
            print("Updated thred_map in tracking process.")
            background_update_event.set()  # Signal tracking process to update background map


def run_processes(manager, raw_data_queue, point_cloud_queue, background_point_cloud_queue, tracking_result_queue, port, bar_file_path, data_reporting_interval, background_data_generating_time = 30, background_update_interval = 30):
    """
    Runs the processes including real-time tracking and periodic background updating.
    background_data_generating_time (sec): Time in seconds to generate a background map.
    background_update_interval (sec): Time in seconds to update the background map.
    
    """
    try:
        # Step 1: **Initial Background Data Generation**
        free_udp_port(port)
        # print("Starting initial background data collection...")

        packet_reader_process = multiprocessing.Process(target=read_packets_online, args=(port, raw_data_queue,))
        packet_parser_process = multiprocessing.Process(target=parse_packets, args=(raw_data_queue,background_point_cloud_queue,))

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
            Td_map = safe_queue_get(background_point_cloud_queue, default=None)
            aggregated_maps.append(Td_map)

        aggregated_maps = np.array(aggregated_maps)
        # print("Generating initial background map...")
        initial_thred_map = gen_bckmap(aggregated_maps, N=10, d_thred=0.1, bck_n=3)

        # Clear queues instead of redefining them
        clear_queue(raw_data_queue)
        clear_queue(background_point_cloud_queue)
        free_udp_port(port)

        # print("Starting real-time monitoring...")

        # Step 2: **Real-Time Tracking**
        tracking_param_update_event = manager.Event()
        tracking_process_stop_event = manager.Event()
        background_update_event = manager.Event()
        background_point_copy_event = manager.Event() # copy point cloud from end of parsing process to background_point_cloud_queue
        tracking_parameter_dict = {
        'win_width': 11,
        'win_height': 7,
        'eps': 1.2,
        'min_samples': 10,
        'missing_thred': 5,
        'bck_radius': 0.2,
        'N' : 10,
        'd_thred' : 0.1,
        "bck_n" : 3
    }
        # config = Config()
        tracking_parameter_dict['win_size'] = [
            tracking_parameter_dict['win_width'],
            tracking_parameter_dict['win_height']
        ]
        tracking_parameter_dict = manager.dict(tracking_parameter_dict)

        bar_drawer = BarDrawer(bar_file_path=bar_file_path)

        # Shared dictionary for thred_map updates
        thred_map_dict = manager.dict({"thred_map": initial_thred_map})
        tracking_pool_dict = manager.dict({})
        off_tracking_pool_dict = manager.dict({})
        mot = MOT(tracking_parameter_dict, thred_map_dict["thred_map"], missing_thred=10)

        # Creating processes
        packet_reader_process = multiprocessing.Process(target=read_packets_online, name= 'ReadPacket', args=(port, raw_data_queue,))
        
        packet_parser_process = multiprocessing.Process(target=parse_packets, name= 'ParsePacket', args=(raw_data_queue, point_cloud_queue,background_point_cloud_queue,background_point_copy_event,))
        tracking_process = multiprocessing.Process(target=track_point_clouds,name= 'Tracking', args=(
            tracking_process_stop_event, mot, point_cloud_queue,
            tracking_parameter_dict, tracking_param_update_event,
            background_update_event,thred_map_dict,bar_drawer
        ))
        traffic_stats_process = multiprocessing.Process(target=count_traffic_stats,name= 'Functional', args=(
            tracking_result_queue, tracking_pool_dict ,bar_drawer, data_reporting_interval
        ))
        background_update_proc = multiprocessing.Process(target=background_update_process, name= 'BackgroundUpdate', args=(
            thred_map_dict,background_point_copy_event ,background_point_cloud_queue, background_update_interval,background_data_generating_time,background_update_event,  # Update every 10 minutes
        ))

        # **Start Queue Monitoring Process**
        queue_monitor_proc = multiprocessing.Process(target=queue_monitor_process, name= 'QueueMonitor', args=(
            raw_data_queue, point_cloud_queue, tracking_result_queue, 5000, 5  # Max size = 5000, Check every 10s
        ))
        
        process_list = [tracking_process, traffic_stats_process, packet_reader_process, packet_parser_process, background_update_proc,queue_monitor_proc]
        
        # Start processes
        for proc in process_list:
            proc.start()
        health_monitor_proc = multiprocessing.Process(target=process_health_monitor, name= 'HealthMonitor' , args=(process_list,))
        health_monitor_proc.start()
        print("Processes started!")
        # Wait for termination signal
        while True:
            try:
                pass  # Keep running
            except KeyboardInterrupt:
                print("Shutting down processes...")
                tracking_process_stop_event.set()
                # Cleanup
                for proc in process_list:
                    proc.terminate()
                    proc.join()
                print("Multiprocessing test complete!")
                break

    except KeyboardInterrupt as e:
        print(e)
        print("Shutting down processes...")
        tracking_process_stop_event.set()  # Signal processes to stop cleanly
        # Cleanup
        for proc in process_list:
            proc.terminate()
            proc.join()
        health_monitor_proc.terminate()
        health_monitor_proc.join()
        
        print("Multiprocessing test complete!")

if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn")
    # 120 cycle length in Artemisia
    bar_file_path = r'./bars.txt'
    port = 2380
    mode = "online" 
    data_reporting_interval = 5 # min
    background_data_generting_time = 30 # sec used frames to generate background
    background_update_interval = 360 # sec
    with multiprocessing.Manager() as manager:
        # Define queues **once** and reuse them
        raw_data_queue = manager.Queue()
        point_cloud_queue = manager.Queue()
        tracking_result_queue = manager.Queue()
        background_point_cloud_queue = manager.Queue()
        run_processes(manager, raw_data_queue, point_cloud_queue, background_point_cloud_queue, tracking_result_queue, port, bar_file_path, data_reporting_interval, background_data_generting_time,background_update_interval)
