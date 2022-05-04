import argparse
from tqdm import tqdm
from matplotlib import cm
from VisulizerTools import *
from BfTableGenerator import TDmapLoader
import matplotlib.pyplot as plt
import os

start_frame = 0
end_frame = 310
pcap_path =  r'D:\LiDAR_Data\ASWS\Thomas\2022-04-06-15-11-07.pcap'
output_path = r'D:\Test\vis_traj'
input_path = r'D:\Test\test.csv'
aggregated_map = []
lidar_reader = TDmapLoader(pcap_path)
frame_gen = lidar_reader.frame_gen()
for i in range(3):
    Frame = next(frame_gen)
    if Frame is None:
        break 
    Td_map,_ = Frame
    aggregated_map.append(Td_map)
aggregated_map = np.array(aggregated_map)

pcd = get_pcd_colored(aggregated_map[-1],np.ones_like(aggregated_map[-1]))
points = np.asarray(pcd.points)

data = pd.read_csv(input_path)
appearance_dic = {}
for f in tqdm(range(start_frame,end_frame)):
    coords = np.array(data.loc[data.FrameIndex == f].loc[:,['Coord_X','Coord_Y']])
    ObjID = np.array(data.loc[data.FrameIndex == f].loc[:,'ObjectID']).astype('int')
    classes = np.array(data.loc[data.FrameIndex == f].loc[:,'Class']).astype('int')
    for i,objid in enumerate(ObjID):
        if objid not in appearance_dic.keys():
            appearance_dic[objid] = [coords[i]]
        else:
            appearance_dic[objid].append(coords[i])
    plt.figure(figsize = (20,20))
    plt.scatter(points[:,0],points[:,1],s = 3,alpha = 0.5 , c = 'gray')
    for key in appearance_dic.keys():
        coord_key = np.array(appearance_dic[key])
        plt.plot(coord_key[:,0],coord_key[:,1], marker = 'x', markersize = 5,c = np.array(cm.tab10_r(key%10)).reshape(1,-1))
        clas = classes[ObjID == key]
        plt.annotate('{},{}'.format(key,clas),coord_key[-1],fontsize=20)
    
    plt.savefig(os.path.join(output_path,'{}.png'.format(f)))
    plt.close()