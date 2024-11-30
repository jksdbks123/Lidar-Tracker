import os
import numpy as np
import pandas as pd
from Utils.SaveTrajectoryTools import generate_T,convert_LLH
from p_tqdm import p_umap
from functools import partial

def run_georef(traj_path,output_path,T):
    traj = pd.read_csv(traj_path)
    temp_xyz = np.array(traj.loc[:,['Coord_X','Coord_Y','Coord_Z']])
    normal_ind = []
    for i,e in enumerate(temp_xyz):
        try:
            temp_xyz[i] = e.astype(np.float64)
            normal_ind.append(i)
        except:
            pass 
    temp_xyz = temp_xyz[normal_ind].astype(np.float64)
    traj = traj.iloc[normal_ind].reset_index(drop = True)
    LLH = convert_LLH(temp_xyz,T)
    traj.loc[:,['Longitude','Latitude','Elevation']] = LLH
    traj.to_csv(output_path,index=False)

def CreateBatchGeoRef(self):
    input_path = self.TrajPathEntry_Tab3.get()
    dir_lis = os.listdir(input_path)
    output_folder_path = self.OutputEntry_Tab3.get()
    csv_names = []
    for f in dir_lis:
        if 'csv' in f.split('.'):
            csv_names.append(f)
    if len(csv_names) == 0:
        print('No Trajs in Folder')
    ref_LLH_path,ref_xyz_path = self.RefLlhEntry_Tab3.get(),self.RefXyzEntry_Tab3.get()
    ref_LLH,ref_xyz = np.array(pd.read_csv(ref_LLH_path)),np.array(pd.read_csv(ref_xyz_path))
    if len(np.unique(ref_xyz[:,2])) == 1:
        np.random.seed(1)
        offset = np.random.normal(-0.521,3.28,len(ref_LLH))
        ref_xyz[:,2] += offset
        ref_LLH[:,2] += offset * 3.2808
    ref_LLH[:,[0,1]] = ref_LLH[:,[0,1]] * np.pi/180
    ref_LLH[:,2] = ref_LLH[:,2]/3.2808
    print(ref_LLH)
    print(ref_xyz)
    T = generate_T(ref_LLH,ref_xyz)
    output_folder_path = os.path.join(output_folder_path,'ProcessedTrajs')
    if os.path.exists(output_folder_path) is not True:
        os.makedirs(output_folder_path)
    n_cpu = self.cpu_nTab2.get()
    traj_paths = [os.path.join(input_path,f) for f in csv_names]
    out_paths = [os.path.join(output_folder_path,f) for f in csv_names]
    print('Begin Geo Referencing')
    p_umap(partial(run_georef,T = T), traj_paths,out_paths,num_cpus = n_cpu)