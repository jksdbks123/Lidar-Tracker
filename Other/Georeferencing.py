import argparse
from random import seed
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
a = 6378137
b = 6356752.31414
e1=(a**2-b**2)/(a**2) #First eccentricity of the ellipsoid
e2=(a**2-b**2)/(b**2) #second eccentricity of the ellipsoid

def generate_T(ref_LLF,ref_xyz):# generate nec T for coord transformation 
    A_ = np.concatenate([ref_xyz,np.ones(ref_xyz.shape[0]).reshape(-1,1)],axis = 1)
    temp = ref_LLF
    N = a/np.sqrt((1 - e1 * np.sin(temp[:,0])**2)) 
    B = np.concatenate([
        ((N + temp[:,2]) * np.cos(temp[:,0]) * np.cos(temp[:,1])).reshape(-1,1),
    ((N + temp[:,2]) * np.cos(temp[:,0]) * np.sin(temp[:,1])).reshape(-1,1),
    ((N*(1 - e1) + temp[:,2]) * np.sin(temp[:,0])).reshape(-1,1)
    ],axis = 1)
    B = np.concatenate([B,np.ones(B.shape[0]).reshape(-1,1)],axis = 1)
    T = np.linalg.inv((A_.T).dot(A_)).dot(A_.T.dot(B))
    return T

def convert_LLH(xyz,T):
    xyz1 = np.concatenate([xyz,np.ones(len(xyz)).reshape(-1,1)],axis = 1)
    XYZ1 = xyz1.dot(T)
    lon = (np.arctan(XYZ1[:,1]/XYZ1[:,0])-np.pi)*180/np.pi
    theta = np.arctan(a * XYZ1[:,2]/(b * np.sqrt(XYZ1[:,0]**2 + XYZ1[:,1]**2)))
    lat = np.arctan((XYZ1[:,2] + e2*b*np.sin(theta)**3)/(np.sqrt(XYZ1[:,0]**2 + XYZ1[:,1]**2) - e1*a*np.cos(theta)**3))
    evel = XYZ1[:,2]/np.sin(lat) - (a/np.sqrt((1 - e1 * np.sin(lat)**2)))*(1 - e1)
    lat = lat*180/np.pi
    LLH = np.concatenate([lon.reshape(-1,1),lat.reshape(-1,1),evel.reshape(-1,1)],axis = 1)
    return LLH

parser = argparse.ArgumentParser(description='This is a program generating georeferencing trajs')
parser.add_argument('-i','--input', help='path that contains .csv file', required=True)
parser.add_argument('-o','--output', help='specified output path', required=True)
args = parser.parse_args()
input_path = args.input
dir_lis = os.listdir(input_path)
output_file_path = args.output
csv_paths = []
output_file_paths = []

for f in dir_lis:
    if ('csv' in f.split('.'))&(f.split('.')[0] != 'LLE_ref')&(f.split('.')[0] != 'xyz_ref'):
        csv_paths.append(f)

if len(csv_paths) == 0:
    print('Traj file is not detected')

ref_LLH_path,ref_xyz_path = os.path.join(input_path,'LLE_ref.csv'),os.path.join(input_path,'xyz_ref.csv')
ref_LLH,ref_xyz = np.array(pd.read_csv(ref_LLH_path)),np.array(pd.read_csv(ref_xyz_path))
if len(np.unique(ref_xyz[:,2])) == 1:
    np.random.seed(1)
    offset = np.random.normal(-0.521,3.28,len(ref_LLH))
    ref_xyz[:,2] += offset
    ref_LLH[:,2] += offset
ref_LLH[:,[0,1]] = ref_LLH[:,[0,1]] * np.pi/180
ref_LLH[:,2] = ref_LLH[:,2]/3.2808
ref_LLH = ref_LLH.astype(np.float64)
ref_xyz = ref_xyz.astype(np.float64)

print(ref_LLH)
print(ref_xyz)
T = generate_T(ref_LLH,ref_xyz)

output_file_folder_path = os.path.join(output_file_path,'ProcessedTrajs')
if os.path.exists(output_file_folder_path) is not True:
    os.makedirs(output_file_folder_path)

for i,p in enumerate(tqdm(csv_paths)):
    traj = pd.read_csv(os.path.join(input_path,p))

    temp_xyz = np.array(traj.loc[:,['Coord_x','Coord_y','Coord_z']])
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
    traj.iloc[:,[11,12,13]] = LLH
    traj.to_csv(os.path.join(output_file_folder_path,p),index=False)

