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
    def __init__(self,frame_path):
        self.frame_path = frame_path

        
    def extract_frame_dic(self,point_cloud,Adb):
        label = Adb.fit_predict(point_cloud)
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
        return frame_dic

    def DBSCANframe_generator(self,eps,min_samples):
        db = DBSCAN(eps=eps,min_samples=min_samples)
        while True:
            for i in range(len(self.frame_path)):
                point_cloud = np.array((pd.read_csv(self.frame_path[i])).loc[:,['X','Y']])
                label = db.fit_predict(point_cloud)
                uniq_label = np.unique(label)
                if -1 in uniq_label:
                    uniq_label = uniq_label[uniq_label!=-1]
                frame_dic = {}
                for ind in range(len(uniq_label)):
                    include_point = point_cloud[label == uniq_label[ind]].astype(np.float32)
                    rect = cv2.minAreaRect(include_point)
                    center = rect[0]
                    box_corner = cv2.boxPoints(rect)
                    detected_obj = DetectedObject(np.array(center),include_point,box_corner)
                    frame_dic[ind] = detected_obj
                yield frame_dic 

    def ADBSCANframe_generator(self,beta,min_sample_1,min_sample_2,min_sample_3):
        Adb = AdaptiveDBSCAN(beta,min_sample_1,min_sample_2,min_sample_3)
        while True:
            for i in range(len(self.frame_path)):
                point_cloud = np.array((pd.read_csv(self.frame_path[i])).loc[:,['X','Y','distance_m']])
                frame_dic = self.extract_frame_dic(point_cloud,Adb)
                yield frame_dic 

    def ADBSCAN_pcap_frame_generator(self,beta,min_sample_1,min_sample_2,min_sample_3):
        Adb = AdaptiveDBSCAN(beta,min_sample_1,min_sample_2,min_sample_3)
        lidar_reader = LidarLoader(self.frame_path)
        frame_gen = lidar_reader.frame_gen()
        while True:
            while True:
                point_cloud = next(frame_gen)[:,[1,2,4,3,5]] # X,Y,D,Z,I
                frame_dic = self.extract_frame_dic(point_cloud,Adb)
                yield frame_dic
            
            

        

if __name__ == "__main__":    
    os.chdir(r'/Users/czhui960/Documents/Lidar/to ZHIHUI/US 395')
    file_path  = os.listdir()[-4]
    beta = 12
    min_sample_1 = 19657
    min_sample_2 = -2.138
    min_sample_3 = -100
    test = FrameGen(file_path).ADBSCAN_pcap_frame_generator(beta,min_sample_1,min_sample_2,min_sample_3)
    print(next(test).keys())
    # print(next(test).keys())
