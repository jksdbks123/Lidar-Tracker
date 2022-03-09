from mimetypes import init
import numpy as np
from multiprocessing import Array
from multiprocessing import Process
import time
from multiprocessing import shared_memory
from multiprocessing import Manager
from BfTableGenerator import *
import sys

 

def reading_from_port(Td_map_queue,Td_map_queue_bck,pcap_path):
    frame_gen = TDmapLoader(pcap_path).frame_gen()
    while True:
        
        frame = next(frame_gen)
        Td_map_queue.put(frame)
        Td_map_queue_bck.put(frame)

def process_frame(Td_map_queue):
    while True:
        time.sleep(0.01)
        print(Td_map_queue.qsize(),'Queue size for temp')
        # process a frame
        Td_map_queue.get()

def gen_bck(Td_map_queue_bck):
    
    while True:
        time.sleep(8) #use 8 sec to process bck map
        print(Td_map_queue_bck.qsize(),'Background Frames')
        Used_frames = []
        while not Td_map_queue_bck.empty():
            Used_frames.append(Td_map_queue_bck.get())
        print(Used_frames[0])

if __name__ == "__main__":
    # manager = Manager()
    # Td_map_queue = manager.Queue(100)
    # Td_map_queue_bck = manager.Queue(200)
    # pcap_path = r'D:/LiDAR_Data/MidTown/Thoma/2021-12-18-12-0-0.pcap'

    # p1 = Process(target = reading_from_port, args = [Td_map_queue,Td_map_queue_bck,pcap_path])
    # p2 = Process(target = process_frame, args = [Td_map_queue,])
    # p3 = Process(target = gen_bck, args = [Td_map_queue_bck,])
    # p1.start()
    # p2.start()
    # p3.start()
    # p1.join()
    # p2.join()
    # p3.join()
    for i in range(0,2000):
        sys.stdout.write('\rnumber is {},{}'.format(i,i+2))
        sys.stdout.flush()
        time.sleep(1)