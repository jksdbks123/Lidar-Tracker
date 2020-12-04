import imageio
import numpy as np
from matplotlib import animation, cm
from matplotlib import pyplot as plt
from DataReader import LidarLoader
import open3d as op3
import os
from tqdm import tqdm
def get_color(key):
    c_ind = int(key%len(cm.tab20.colors))
    c = cm.tab20(c_ind) 
    return np.array(c)
def visualize_single_obj(detected_obj):
    pass
def generate_background_voxel(resolution, freq_factor, iter_n, file_path, save_path, detecting_range):
    loader = LidarLoader(file_path,0,with_bf=False).frame_gen()
    frame = next(loader)
    grid_width = detecting_range*2
    grid_length = detecting_range*2
    bf_frames = iter_n
    bf_threshold = freq_factor
    X_coordinates = np.arange(-grid_width/2,grid_width/2,resolution)
    Y_coordinates = np.arange(-grid_length/2,grid_length/2,resolution)
    Z_coordinates = np.arange(frame[:,3].min(),frame[:,3].max(),resolution)
    X, Y ,Z = np.meshgrid(X_coordinates, Y_coordinates,Z_coordinates)
    grid_3d_points = np.vstack((X,Y,Z)).reshape(3,-1).T
    print('Creating VoxelGrids..')
    pcd_total_voxel = op3.geometry.PointCloud()
    pcd_total_voxel.points = op3.utility.Vector3dVector(grid_3d_points)
    voxel_grid = op3.geometry.VoxelGrid.create_from_point_cloud(pcd_total_voxel,voxel_size=resolution)
    total_counts = np.zeros((len(X_coordinates),len(Y_coordinates),len(Z_coordinates)))
    for frame in tqdm(range(bf_frames)):
        counts = np.zeros((len(X_coordinates),len(Y_coordinates),len(Z_coordinates)))
        frame = next(loader)
        frame_xyz = frame[:,[1,2,3]]
        for i in range(len(frame_xyz)):
            voxel_cors = voxel_grid.get_voxel(frame_xyz[i])
            try:
                if counts[voxel_cors[0],voxel_cors[1],voxel_cors[2]]==0:
                    counts[voxel_cors[0],voxel_cors[1],voxel_cors[2]]+=1
            except:
                continue
        total_counts = total_counts +counts
    total_counts = total_counts/bf_frames
    Background_X,Background_Y,Background_Z = np.where(total_counts>bf_threshold)
    Background_X = X_coordinates[Background_X]
    Background_Y = Y_coordinates[Background_Y]
    Background_Z = Z_coordinates[Background_Z]
    Background_XYZ = np.concatenate([Background_X.reshape(-1,1),Background_Y.reshape(-1,1),Background_Z.reshape(-1,1)],axis = 1)
    pcd_bck = op3.geometry.PointCloud()
    pcd_bck.points = op3.utility.Vector3dVector(Background_XYZ)
    voxel_grid_bck = op3.geometry.VoxelGrid.create_from_point_cloud(pcd_bck,voxel_size=resolution)
    print(os.path.join(save_path,'bck_voxel.ply'))
    op3.io.write_voxel_grid(os.path.join(save_path,'bck_voxel.ply'),voxel_grid_bck)
    