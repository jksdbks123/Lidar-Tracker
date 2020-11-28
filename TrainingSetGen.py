from MultiTrackingSystem import MultiTrackingSystem
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.animation as animation
import cv2
import open3d as op3

class TrainingSetGen():
    def __init__(self,file_path,iter_n):
        self.trackings = {}
        self.file_path = file_path
        self.iter_n = iter_n
        self.labels = []
        self.xyz = []
        self.intensities = []
        
        
    def generate_trackings(self):
        alpha = np.pi * (0.2)/180
        beta = 12
        corr = np.sin(alpha/2) * 2
        min_sample_1 = 19657
        min_sample_2 = -2.138
        min_sample_3 = -100
        multi_tracking = MultiTrackingSystem(iter_n = self.iter_n, tolerance= 4,gen_fig= False)
        multi_tracking.fit_adbgen_pcap(self.file_path,beta,min_sample_1,min_sample_2,min_sample_3)
        multi_tracking.batch_tracking()
        self.trackings = multi_tracking.out_of_tracking_list
        
    def show_three_frame_traj(self,pcds,ind):
        start = ind - 1 
        end = ind + 1 
        if start<0:
            start = 0
        if end > (len(pcds)-1):
            end = len(pcds) - 1
        target = pcds[ind]
        pcl,_ = target.compute_convex_hull()# convex hull
        hull_ls = op3.geometry.LineSet.create_from_triangle_mesh(pcl)
        hull_ls.paint_uniform_color((1,0,0))
        op3.visualization.draw_geometries([target,hull_ls,pcds[start],pcds[end]])
        
    def show_all_traj(self,pcds):
        if len(pcds) == 1:
            target = pcds[0]
            pcl,_ = target.compute_convex_hull()# convex hull
            hull_ls = op3.geometry.LineSet.create_from_triangle_mesh(pcl)
            hull_ls.paint_uniform_color((1,0,0))
            op3.visualization.draw_geometries([target,hull_ls])
        else:
            op3.visualization.draw_geometries(pcds)
        
    def is_input_correct(self,cur_input):
        if cur_input == 0 or cur_input == 1 or cur_input == 'n' or cur_input == 'N' or cur_input == 'exit':
            return True
        else:
            return False
    def extract_pcds(self,tracking_list,obj_id):
        pcds = []
        intensities = []
        for i in range(len(tracking_list[obj_id].point_clouds)):
            if type(tracking_list[obj_id].point_clouds[i]) != int:
                xy = tracking_list[obj_id].point_clouds[i]
                z = tracking_list[obj_id].elevation_intensities[i][:,0].reshape(-1,1)
                xyz = np.concatenate([xy,z],axis = 1)
                intensity = tracking_list[obj_id].elevation_intensities[i][:,1]
                intensities.append(intensity)
                pcd = op3.geometry.PointCloud()
                pcd.points = op3.utility.Vector3dVector(xyz)
                pcds.append(pcd)
        return pcds,intensities
        
    def labeling(self):
        cur_object_ind = 0 
        processed_point_clouds_num = 0
        comfirmed_samples = []
        comfirmed_intensities = []
        labels = [] # 0: car, 1: horse
        while True:
            print('Current Object Number:', cur_object_ind)
            print('Processed Point Clouds:', processed_point_clouds_num)
            pcds,intensities = self.extract_pcds(self.trackings,cur_object_ind)
            self.show_all_traj(pcds)
            cur_input = input('Save All Trackings?:')
            correct_input = self.is_input_correct(cur_input)
            if cur_input == 'exit':
                break
            while correct_input:
                correct_input = self.is_input_correct(cur_input)
                print('Wrong Input')
                if correct_input:
                    break
            if cur_input == 0 or cur_input == 1:
                added_samples = 0
                for i in range(len(pcds)):
                    comfirmed_samples.append(pcds[i])
                    comfirmed_intensities.append(intensities[i])
                    added_samples += 1
                    labels.append(cur_input)
                processed_point_clouds_num += added_samples
            else:
                for i in range(len(pcds)):
                    self.show_three_frame_traj(pcds,i)
                    cur_input = input('Save the Labelled Tracking?:')
                    while correct_input:
                        correct_input = self.is_input_correct(cur_input)
                        print('Wrong Input')
                        if correct_input:
                            break
                    if cur_input == 'exit':
                        break
                    if cur_input == 1 or cur_input == 0:
                        comfirmed_samples.append(pcds[i])
                        comfirmed_intensities.append(intensities[i])
                        labels.append(cur_input)
                        processed_point_clouds_num += 1
            cur_object_ind+=1
        self.xyz = comfirmed_samples
        self.intensities = comfirmed_intensities
        self.labels = labels

                
                


        
    
    
if __name__ == "__main__":
    os.chdir(r'/Users/czhui960/Documents/Lidar/to ZHIHUI/USA pkwy')
    file_path  = os.listdir()[4]
    label_sys = TrainingSetGen(file_path,1800)
    label_sys.generate_trackings()
    label_sys.labeling()
    
    
    