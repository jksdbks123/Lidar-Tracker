import os

import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from DetectedObject import DetectedObject


class FrameGen():
    def __init__(self,frame_path,eps,min_samples,folder_path):
        self.frame_path = frame_path
        self.folder_path = folder_path
        self.eps = eps
        self.min_samples = min_samples
        self.db = DBSCAN(eps=self.eps,min_samples=self.min_samples)
        

    def frame_generator(self):
        while True:
            for i in range(len(self.frame_path)):
                point_cloud = np.array(pd.read_csv(self.folder_path.format(self.frame_path[i])).loc[:,['X','Y']])
                label = self.db.fit_predict(point_cloud)
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


if __name__ == "__main__":
    
    folder = r'./frames/2019-8-27-7-0-0-BF1(0-18000frames)/{}'
    os.chdir(r'/Users/czhui960/Documents/Lidar/to ZHIHUI/US 395')
    frames_name = os.listdir(r'./frames/2019-8-27-7-0-0-BF1(0-18000frames)/')
    frames_name.sort(key = lambda x : x.split(' ')[2][:-5])
    test = FrameGen(frames_name,2.2,10,folder).frame_generator()
    print(next(test).keys())
    print(next(test).keys())
