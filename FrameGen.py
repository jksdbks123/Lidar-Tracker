import os
import dpkt
import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from DataReader import LidarLoader
from DetectedObject import DetectedObject
from LidarDBSCAN import AdaptiveDBSCAN

class FrameGen():
    def __init__(self,frame_path,detecting_range,bck_voxel_path,with_bf = True):
        self.frame_path = frame_path
        self.bck_voxel_path = bck_voxel_path
        self.with_bf = with_bf
        self.detecting_range = detecting_range
        
    def extract_frame_dic(self,point_cloud,Adb):
        if len(point_cloud) == 0:
            return {}
        label = Adb.fit_predict(point_cloud)# cluster part point_cloud: 2D array [[2.1,2.3],[2.2,3.4],[x,y],[100,121]] label:[0,0,1,-1]
        uniq_label = np.unique(label)
        if -1 in uniq_label:
            uniq_label = uniq_label[uniq_label!=-1]
        frame_dic = {}
        for ind in range(len(uniq_label)):
            include_point = point_cloud[label == uniq_label[ind]].astype(np.float32)[:,[0,1]]
            rect = cv2.minAreaRect(include_point)
            center = rect[0]
            box_corner = cv2.boxPoints(rect)
            elevation_intensity = point_cloud[label == uniq_label[ind]].astype(np.float32)[:,[3,4]]
            detected_obj = DetectedObject(np.array(center),include_point,box_corner,elevation_intensity)
            frame_dic[ind] = detected_obj

        return frame_dic #next_frame

    def DBSCAN_pcap_frame_generator(self,eps,min_samples):
        db = DBSCAN(eps=eps,min_samples=min_samples)
        lidar_reader = LidarLoader(self.frame_path,self.bck_voxel_path,self.detecting_range,self.with_bf)
        frame_gen = lidar_reader.frame_gen()
        while True:
            while True:
                point_cloud = next(frame_gen)[:,[1,2,4,3,5]] # X,Y,D,Z,I
                frame_dic = self.extract_frame_dic(point_cloud,db)
                yield frame_dic

    def ADBSCAN_pcap_frame_generator(self,beta,min_sample_1,min_sample_2,min_sample_3):
        Adb = AdaptiveDBSCAN(beta,min_sample_1,min_sample_2,min_sample_3)
        lidar_reader = LidarLoader(self.frame_path,self.bck_voxel_path,self.detecting_range,self.with_bf)
        frame_gen = lidar_reader.frame_gen()
        while True:
            while True:
                point_cloud = next(frame_gen)[:,[1,2,4,3,5]] # X,Y,D,Z,I
                frame_dic = self.extract_frame_dic(point_cloud,Adb)
                yield frame_dic
            
            
if __name__ == "__main__":
    pcap_file_path = r'/Users/czhui960/Documents/Lidar/to ZHIHUI/USA pkwy/2019-12-18-10-0-0-BF1(0-18000frames).pcap'
    detecting_range = 40 #meter
    bck_voxel_path = 0
    eps = 1
    min_samples = 7
    frame_gen = FrameGen(pcap_file_path,detecting_range,bck_voxel_path,with_bf=False).DBSCAN_pcap_frame_generator(eps,min_samples)
    for i in range(100):
        next_frame = next(frame_gen)
        if i == 10:
            print(next_frame[0].point_cloud)