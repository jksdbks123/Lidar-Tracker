from MultiTrackingSystem import MultiTrackingSystem
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.animation as animation
import IPython.display as display
import cv2
import open3d as op3

class TrainingSetGen():
    def __init__(self,file_path):
        self.trackings = {}
        self.file_path = file_path
    def generate_trackings(self):
        pass
if __name__ == "__main__":
    alpha = np.pi * (0.2)/180
    beta = 12
    corr = np.sin(alpha/2) * 2
    min_sample_1 = 19657
    min_sample_2 = -2.138
    min_sample_3 = -100
    os.chdir(r'/Users/czhui960/Documents/Lidar/to ZHIHUI/USA pkwy')
    file_path  = os.listdir()[4]
    multi_tracking = MultiTrackingSystem(iter_n = 100, tolerance= 4,gen_fig= False)
    multi_tracking.fit_adbgen_pcap(file_path,beta,min_sample_1,min_sample_2,min_sample_3)
    multi_tracking.batch_tracking()
    pcds = []
    for i in range(len(multi_tracking.out_of_tracking_list[0].point_clouds)):
        if type(multi_tracking.out_of_tracking_list[0].point_clouds[i]) != int:
            xy = multi_tracking.out_of_tracking_list[0].point_clouds[i]
            z = multi_tracking.out_of_tracking_list[0].elevation_intensities[i][:,0].reshape(-1,1)
            xyz = np.concatenate([xy,z],axis = 1)
            intensity = multi_tracking.out_of_tracking_list[0].elevation_intensities[i][:,1]
            pcd = op3.geometry.PointCloud()
            pcd.points = op3.utility.Vector3dVector(xyz)
            pcds.append(pcd)
    test_pcd = pcds[10]
    pcl,_ = test_pcd.compute_convex_hull()# convex hull
    hull_ls = op3.geometry.LineSet.create_from_triangle_mesh(pcl)
    hull_ls.paint_uniform_color((1,0,0))
    op3.visualization.draw_geometries([test_pcd,hull_ls,pcds[9],pcds[11]])