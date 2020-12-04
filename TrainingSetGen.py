from MultiTrackingSystem import MultiTrackingSystem
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import matplotlib.animation as animation
import cv2
import open3d as op3

class TrainingSetGen():
    def __init__(self,file_path,iter_n,bck_voxel_path):
        self.trackings = {}
        self.file_path = file_path
        self.iter_n = iter_n
        self.labels = []
        self.xyz = []
        self.intensities = []
        self.bck = 0
        self.bck_voxel_path = bck_voxel_path
        
    def generate_trackings(self):
        
        pcap_file_path  = r'./2019-12-18-10-0-0-BF1(0-18000frames).pcap'
        bck_path = r'./Results/bck_voxel.ply'
        iter_n = 700
        tolerance = 4
        bf_resolution = 0.2
        detecting_range = 50
        freq_factor = 0.008
        iter_bf = 3000

        multi_tracking = MultiTrackingSystem(iter_n, tolerance,bf_resolution,detecting_range,freq_factor,iter_bf,pcap_file_path,gen_fig= False,BF=False)
        multi_tracking.initial_work_space()
        multi_tracking.fit_dbgen_pcap(3,7)
        multi_tracking.batch_tracking()
        self.trackings = multi_tracking.out_of_tracking_list
        self.bck = op3.io.read_voxel_grid(self.bck_voxel_path)
        
    def show_three_frame_traj(self,pcds,intensities,ind):
        start = ind - 1 
        end = ind + 1 
        if start<0:
            start = 0
        if end > (len(pcds)-1):
            end = len(pcds) - 1
        target_pcd = pcds[ind]
        target_intensity = intensities[ind]
        target_intensity = target_intensity/256
        target_pcd.colors = op3.utility.Vector3dVector(np.tile(target_intensity.reshape((-1,1)),3))
        voxel = op3.geometry.VoxelGrid.create_from_point_cloud(target_pcd,0.1)
        bounding_box = op3.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(target_pcd)
        op3.visualization.draw_geometries([voxel,pcds[start],pcds[end],bounding_box,self.bck])
        
    def show_all_traj(self,pcds,intensities):
        if len(pcds) == 1:
            target_pcd = pcds[0]
            target_intensity = intensities[0]
            target_intensity = target_intensity/256
            target_pcd.colors = op3.utility.Vector3dVector(np.tile(target_intensity.reshape((-1,1)),3))
            voxel = op3.geometry.VoxelGrid.create_from_point_cloud(target_pcd,0.1)
            bounding_box = op3.geometry.AxisAlignedBoundingBox.get_axis_aligned_bounding_box(target_pcd)
            op3.visualization.draw_geometries([voxel,bounding_box,self.bck])
        else:
            pcds.append(self.bck)
            op3.visualization.draw_geometries(pcds)
            
        
    def is_input_correct(self,cur_input):
        inputs = ['0','1','2','n','N','exit']
        if cur_input in inputs:
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
        labels = [] # 0: car, 1: horse, 2: unknown
        while True:
            print('Current Object Number:', cur_object_ind)
            print('Processed Point Clouds:', processed_point_clouds_num)
            pcds,intensities = self.extract_pcds(self.trackings,cur_object_ind)
            self.show_all_traj(pcds,intensities)
            cur_input = input('Save All Trackings?:')
            correct_input = self.is_input_correct(cur_input)
            while ~correct_input:
                correct_input = self.is_input_correct(cur_input)
                print('Wrong Input')
                if correct_input:
                    break
            if cur_input == 'exit':
                break
            pcds.pop()
            if (cur_input == '0') | (cur_input == '1'):
                added_samples = 0
                for i in range(len(pcds)):
                    comfirmed_samples.append(pcds[i])
                    comfirmed_intensities.append(intensities[i])
                    added_samples += 1
                    labels.append(cur_input)
                processed_point_clouds_num += added_samples
            else:
                for i in range(len(pcds)):
                    self.show_three_frame_traj(pcds,intensities,i)
                    cur_input = input('Save the Labelled Tracking?:')
                    while ~correct_input:
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
    bck_path = r'./Results/bck_voxel.ply'
    file_path  = os.listdir()[-1]
    label_sys = TrainingSetGen(file_path,600,bck_path)
    label_sys.generate_trackings()
    label_sys.labeling()
    
    
    