
import numpy as np
from numpy.core.fromnumeric import shape
import open3d as op3
from scipy.optimize.optimize import main
from scipy.spatial import distance
import pandas as pd
from scipy.optimize import linear_sum_assignment
# import torch
from sklearn.cluster import DBSCAN
import cv2
import json
import pickle


class detected_obj():
    def __init__(self):
        self.glb_id = None
        self.start_frame = None
        self.state = None 
        self.apperance = None 
        self.rows = None
        self.cols = None # row, col inds in the Td-map 
        self.label_seq = [] # represented labels at each frame 
        self.mea_seq = []
        self.post_seq = []
        self.app_seq = []
        

#calibration 
theta_raw = np.array([[-25,1.4],[-1,-4.2],[-1.667,1.4],[-15.639,-1.4],
                            [-11.31,1.4],[0,-1.4],[-0.667,4.2],[-8.843,-1.4],
                            [-7.254,1.4],[0.333,-4.2],[-0.333,1.4],[-6.148,-1.4],
                            [-5.333,4.2],[1.333,-1.4],[0.667,4.2],[-4,-1.4],
                            [-4.667,1.4],[1.667,-4.2],[1,1.4],[-3.667,-4.2],
                            [-3.333,4.2],[3.333,-1.4],[2.333,1.4],[-2.667,-1.4],
                            [-3,1.4],[7,-1.4],[4.667,1.4],[-2.333,-4.2],
                            [-2,4.2],[15,-1.4],[10.333,1.4],[-1.333,-1.4]
                            ])[:,0]
theta = np.sort(theta_raw)
azimuths = np.arange(0,360,0.2)
# color map 
np.random.seed(412)
color_map = np.random.random((100,3))

#xylwh xylwh, xy

def convert_point_cloud(Td_map, Labeling_map, Thred_map): 
    td_freq_map = Td_map
    Xs = []
    Ys = []
    Zs = []
    Labels = []
    for i in range(td_freq_map.shape[0]):

        longitudes = theta[i]*np.pi / 180
        latitudes = azimuths * np.pi / 180 
        hypotenuses = td_freq_map[i] * np.cos(longitudes)
        X = hypotenuses * np.sin(latitudes)
        Y = hypotenuses * np.cos(latitudes)
        Z = td_freq_map[i] * np.sin(longitudes)
        Valid_ind =  (td_freq_map[i] != 0)&(td_freq_map[i]<Thred_map[i])

        # None zero index
        Xs.append(X[Valid_ind])
        Ys.append(Y[Valid_ind])
        Zs.append(Z[Valid_ind])

        Labels.append(Labeling_map[i][Valid_ind])

    Xs = np.concatenate(Xs)
    Ys = np.concatenate(Ys)
    Zs = np.concatenate(Zs)
    Labels = np.concatenate(Labels).astype('int')
    XYZ = np.concatenate([Xs.reshape(-1,1),Ys.reshape(-1,1),Zs.reshape(-1,1)],axis = 1)
    return XYZ, Labels

def get_params_from_detection_points(point):
    
    pcd = op3.geometry.PointCloud()
    pcd.points = op3.utility.Vector3dVector(point)
    bbox = pcd.get_axis_aligned_bounding_box()
    # x,y,length,width,height 
    xylwh = np.concatenate([bbox.get_center()[:2],bbox.get_max_bound() - bbox.get_min_bound()])
    return xylwh

def count(TSAv):
    temp_count = 0
    apear_ind = []
    counts = []
    for i in range(len(TSAv)):
        if (TSAv[i] == True):
            temp_count += 1
        else:
            if (i > 0) & (TSAv[i - 1] == True):
                apear_ind.append(i - temp_count)
                counts.append(temp_count)
                temp_count = 0
                counts.append(0)
            else:
                counts.append(0)
        if (i == len(TSAv) - 1) & (temp_count != 0):
            apear_ind.append(i - temp_count + 1)
            counts.append(temp_count)
    counts = np.array(counts)
    counts = counts[counts > 0]
    return np.array(counts), np.array(apear_ind)

def if_bck(rows,cols,Td_map,Plane_model):
    # check if an object is background
    # car: 2.6m 
    td_freq_map = Td_map
    longitudes = theta[rows]*np.pi / 180
    latitudes = azimuths[cols] * np.pi / 180 
    hypotenuses = td_freq_map[rows,cols] * np.cos(longitudes)
    X = hypotenuses * np.sin(latitudes)
    Y = hypotenuses * np.cos(latitudes)
    Z = td_freq_map[rows,cols] * np.sin(longitudes)
    Height_from_ground = Plane_model[0] * X + Plane_model[1] * Y  + Plane_model[2] * Z + Plane_model[3] 
    Max_Height = Height_from_ground.max()
    if (Max_Height > 3)|(Max_Height < 0.3):
        return True
    else:
        return False   

def get_pcd_uncolored(Td_map):

    Xs = []
    Ys = []
    Zs = []
    for i in range(Td_map.shape[0]):
        longitudes = theta[i]*np.pi / 180
        latitudes = azimuths * np.pi / 180 
        hypotenuses = Td_map[i] * np.cos(longitudes)
        X = hypotenuses * np.sin(latitudes)
        Y = hypotenuses * np.cos(latitudes)
        Z = Td_map[i] * np.sin(longitudes)
        Xs.append(X)
        Ys.append(Y)
        Zs.append(Z)

    Xs = np.concatenate(Xs)
    Ys = np.concatenate(Ys)
    Zs = np.concatenate(Zs)
    pcd = op3.geometry.PointCloud()
    XYZ = np.concatenate([Xs.reshape(-1,1),Ys.reshape(-1,1),Zs.reshape(-1,1)],axis = 1)
    pcd.points = op3.utility.Vector3dVector(XYZ)
    return pcd    

def extract_xy(Labeling_map,Td_map):
        
    # Plane_model is a 1 x 4 array representing a,b,c,d in ax + by + cz + d = 0 
    new_uni_labels = np.unique(Labeling_map)
        #Only Background contains 
    if -1 in new_uni_labels:
        new_uni_labels = new_uni_labels[1:]
    xy_set = [] # xy position and apperance features
    apperance_set = []
    for label in new_uni_labels:
        rows,cols = np.where(Labeling_map == label)

        rows_temp,cols_temp = rows.copy(),cols.copy()
        sort_ind = np.argsort(cols)
        refer_cols = cols[sort_ind[[0,-1]]]
        # this is being said, the first place is for less azimuth id 
        refer_rows = rows[sort_ind[[0,-1]]]
        if np.abs(refer_cols[0] - refer_cols[1]) >= 900:
            cols[cols <= 900] += 1800
            sort_ind = np.argsort(cols)
            refer_cols = cols[sort_ind[[0,-1]]]
            refer_cols[refer_cols >= 1800] -= 1800
            refer_rows = rows[sort_ind[[0,-1]]]
        apperance = get_appearance_features(rows_temp,cols_temp,Td_map)
        xy = get_representative_point(refer_rows,refer_cols,Td_map) # x,y vec for two representatives 
        xy_set.append(xy)
        apperance_set.append(apperance)
    # apperance is a 1 x 8 x 1 vec including:  dis, point_cnt, dir_vec_x, dir_vec_y, height, length, width 
    # x , y is 2 x 2 x 1
    xy_set = np.array(xy_set)
    # n x 2 x 2 x 1
    apperance_set = np.array(apperance_set)
    new_uni_labels = np.unique(Labeling_map)
    if -1 in new_uni_labels:
        new_uni_labels = new_uni_labels[1:]

    return xy_set,apperance_set,new_uni_labels,Labeling_map



def get_appearance_features(rows,cols,Td_map): #obtain length height and width
    
    # 1.dis 2. point cnt 3.Dir(x) 4.dir(y) 5.Height 6.Len 7.Width 8.2Darea

    td_freq_map = Td_map
    dis = td_freq_map[rows,cols]
    longitudes = theta[rows]*np.pi / 180
    latitudes = azimuths[cols] * np.pi / 180 
    hypotenuses = dis * np.cos(longitudes)
    X = hypotenuses * np.sin(latitudes)
    Y = hypotenuses * np.cos(latitudes)
    Z = dis * np.sin(longitudes)
    points = np.array([X,Y]).T
    points_num = len(points)
    rect = cv2.minAreaRect(points.astype('float32'))
    box = cv2.boxPoints(rect)
    # box = cv2.boxPoints(rect)
    b1 = np.sqrt(np.sum((box[1] - box[0])**2))
    b2 = np.sqrt(np.sum((box[2] - box[1])**2))
    length = b1
    width = b2
    dir_vec = box[1] - box[0]
    if b1 < b2:
        length = b2
        width = b1
        dir_vec = box[2] - box[1]
    modility = np.sqrt(np.sum(dir_vec**2))
    if modility == 0:
        dir_vec = np.array([0,0])
    else:
        dir_vec = dir_vec/modility
    height = Z.max() - Z.min()
    area = length * width
    vec = np.array([points_num,dir_vec[0],dir_vec[1],height,length,width,area,dis.mean()]).reshape(-1,1)
    # vec = np.full((2,8,1),vec) # status vec for two representative points 
    return vec #1 x 8 x 1  

def get_representative_point(ref_rows,ref_cols,Td_map): 
    td_freq_map = Td_map
    longitudes = theta[ref_rows]*np.pi / 180
    latitudes = azimuths[ref_cols] * np.pi / 180 
    hypotenuses = td_freq_map[ref_rows,ref_cols] * np.cos(longitudes)
    X = hypotenuses * np.sin(latitudes)
    Y = hypotenuses * np.cos(latitudes)
    # Z = td_freq_map[ref_rows,ref_cols] * np.sin(longitudes)
    
    return np.array([
        [X[0],Y[0]],
        [X[1],Y[1]]
    ]).reshape(2,2,1) # n_repr x xy_dim x 1 


    

def linear_assignment(State_affinity):

    associated_ind_cur,associated_ind_next = [],[]
    associated_ind_cur_extend_,associated_ind_next_extend_= linear_sum_assignment(State_affinity,maximize = True)
    for i in range(len(associated_ind_cur_extend_)):
        if State_affinity[associated_ind_cur_extend_[i],associated_ind_next_extend_[i]] != 0:
            associated_ind_cur.append(associated_ind_cur_extend_[i])
            associated_ind_next.append(associated_ind_next_extend_[i])
    associated_ind_cur,associated_ind_next = np.array(associated_ind_cur),np.array(associated_ind_next)
    ind = np.argsort(associated_ind_cur)
    associated_ind_cur = associated_ind_cur[ind]
    associated_ind_next = associated_ind_next[ind]

    return associated_ind_cur,associated_ind_next




def create_new_detection(Tracking_pool,Global_id,state_init,app_init,label_init,mea_init,start_frame):
    new_detection = detected_obj()
    new_detection.glb_id = Global_id
    new_detection.state = state_init
    new_detection.apperance = app_init
    new_detection.label_seq.append(label_init)
    new_detection.start_frame = start_frame
    new_detection.mea_seq.append(mea_init)
    new_detection.post_seq.append(state_init)
    new_detection.app_seq.append(app_init)
    Tracking_pool[Global_id] = new_detection
        


def process_fails(Tracking_pool,Off_tracking_pool,glb_id):

    Off_tracking_pool[glb_id] = Tracking_pool.pop(glb_id)
        
def associate_detections(Tracking_pool,glb_id,state,app,next_label,mea_next):
    
    Tracking_pool[glb_id].state = state
    Tracking_pool[glb_id].apperance = app
    Tracking_pool[glb_id].label_seq.append(next_label)
    Tracking_pool[glb_id].mea_seq.append(mea_next)
    Tracking_pool[glb_id].post_seq.append(state)
    Tracking_pool[glb_id].app_seq.append(app)

    
def state_update(A,H,state_,P_,R,mea):
    """
    mea: m_k (m x 5 x 1)
    
    """
    K = np.matmul(np.matmul(P_,H.transpose()),np.linalg.inv(np.matmul(np.matmul(H,P_),H.transpose()) + R))
    P = np.matmul((np.eye(A.shape[0]) - np.matmul(K,H)),P_)
    residual = mea - np.matmul(H,state_) # n x 5 x 1
    state = state_ + np.matmul(K,residual)
    
    return state, P 

def get_ovlp_pairs(Labeling_map_cur,Labeling_map_next):
    cooresponding_map = np.array([Labeling_map_cur.flatten(),Labeling_map_next.flatten()]).T
    pairs,counts = np.unique(cooresponding_map,return_counts = True, axis = 0)
    return pairs,counts

def get_affinity_mat_td(app_cur,app_next,unique_label_next,unique_label_cur,Labeling_map_cur,Labeling_map_next):
    
    pairs,counts = get_ovlp_pairs(Labeling_map_cur,Labeling_map_next)
    associated_matrix = np.zeros((app_cur.shape[0],app_next.shape[0]))
    dis_matrix = np.ones((app_cur.shape[0],app_next.shape[0]))
    
    for i,pair in enumerate(pairs):
        if (-1 == pair).any():
            continue
        c = counts[i]
        if c > 500:
            c = 500  

        ind_cur = np.where(unique_label_cur == pair[0])[0][0]
        ind_next = np.where(unique_label_next == pair[1])[0][0]
        associated_matrix[ind_cur,ind_next] = c/500        
        dis = np.abs(app_next[ind_next,-1,0] - app_cur[ind_cur,-1,0])
        if dis > 2:
            dis = 2
        dis_matrix[ind_cur,ind_next] = dis/2

    
    return  0.6*associated_matrix + 0.4*(1 - dis_matrix)
    
        




#sum file
col_names_ = ['X_Coord_est','Y_Coord_est','X_Len_est','Y_Len_est','Z_Len_est','X_Vel_est','Y_Vel_est','X_Acc_est','Y_Acc_est']
col_names = ['X_Coord_mea','Y_Coord_mea','X_Len_mea','Y_Len_mea','Z_Len_mea']

def get_summary_file_split(post_seq,mea_seq):
    temp = np.array(post_seq)
    temp = temp.reshape(temp.shape[0],temp.shape[1])[:,[0,1,2,3,4,5,6,10,11]]
    df_ = pd.DataFrame(temp,columns= col_names_)
    temp = mea_seq
    emp = []
    for i,vec in enumerate(temp):
        if type(vec) == int:
            emp.append(-np.ones(len(col_names)).astype(np.int8))
        else:
            emp.append(vec.flatten())
    emp = np.array(emp)
    df = pd.DataFrame(emp,columns = col_names)
    summary = pd.concat([df,df_],axis = 1)
    return summary


column_names_TR_2o = ['ObjectID','FrameIndex','Coord_X','Coord_Y','Coord_Z','Distance','Speed_X','Speed_Y','Speed(m/s)''Longitude','Latitude','Elevation']
app_col_names = ['Point_Cnt','Dir_X_Bbox','Dir_Y_Bbox','Height','Length','Width','Area']
    # obj_id,ts,x,y,z,d,s_x,s_y,s,L,L,H

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
column_names_TR_2o = ['ObjectID','FrameIndex','Coord_X','Coord_Y','Coord_Z','Distance','Speed_X','Speed_Y','Speed(m/s)','Longitude','Latitude','Elevation']

def get_summary_file_TR(post_seq,key,start_frame,app_seq,T):
    temp = np.array(post_seq)
    temp = temp.reshape((temp.shape[0],temp.shape[1],temp.shape[2]))
    # n x 2 x 6
    temp_xy = temp[:,:,:2]
    # n x 2 x 2
    dis_est = np.sqrt((temp_xy[:,:,0]**2 + temp_xy[:,:,1]**2))
    # n x 2 
    speed_xy = temp[:,:,2:4] * 10 
    # n x 2 x 2
    speed = np.sqrt((speed_xy[:,:,0]**2 + speed_xy[:,:,1]**2))
    # n x 2
    xyz_0 = np.concatenate([temp_xy[:,0],np.zeros(len(temp_xy)).reshape(-1,1)],axis = 1)
    xyz_1 = np.concatenate([temp_xy[:,1],np.zeros(len(temp_xy)).reshape(-1,1)],axis = 1)
    xyz = (xyz_0 + xyz_1)/2
    LLH_est = convert_LLH(xyz,T)
    est = np.concatenate([xyz_0,dis_est[:,0].reshape(-1,1),speed_xy[:,0],speed[:,0].reshape(-1,1),LLH_est],axis = 1)
    # x,y,z,d,s_x,s_y,s,L,L,H
    timestp = []
    for i in range(len(temp)):
        f = i + start_frame + 1
        timestp.append('%06.0f'%f)
    timestp = np.array(timestp).reshape(-1,1)
    objid = (np.ones(len(temp)) * key).astype(int).reshape(-1,1)
    summary = np.concatenate([objid,timestp,est],axis = 1)
    # obj_id,ts,x,y,z,d,s_x,s_y,s,L,L,H
    summary = pd.DataFrame(summary,columns = column_names_TR_2o)

    emp = []
    for app in app_seq:
        if type(app) == int:
            emp_row = np.empty(len(app_col_names))
            emp_row[:] = np.nan
            emp.append(emp_row)
        else:
            emp.append(app.flatten())

    app_df = pd.DataFrame(emp,columns = app_col_names)
    app_df.Length = app_df.Length.max()

    return summary,app_df

def get_summary_file(post_seq,mea_seq,key,start_frame,missing_thred,T):
    
    temp = np.array(post_seq)
    temp = temp.reshape(temp.shape[0],temp.shape[1])[1:-missing_thred] # exclude first and ending data point 
    # [0,1,2,3,4,5,6,10,11]
    temp_lwhxy = temp[:,[2,3,4,0,1]]
    dis_est = np.sqrt(np.sum(temp_lwhxy[:,[3,4]]**2,axis = 1)).reshape(-1,1)
    speed_xy = temp[:,[5,6]]*10  #m/s
    speed = np.sqrt(np.sum(speed_xy**2,axis = 1)).reshape(-1,1)*3600/1000
    #since we don't track the z value, literally use 0 to alternate the z
    xyz = np.concatenate([temp_lwhxy[:,[3,4]],np.zeros(temp_lwhxy.shape[0]).reshape(-1,1)],axis = 1) 
    LLH_est = convert_LLH(xyz,T)
    est = np.concatenate([temp_lwhxy,LLH_est,dis_est,speed_xy,speed],axis = 1)
    temp = mea_seq
    emp = []
    for i,vec in enumerate(temp):
        if type(vec) == int:
            emp_row = np.empty(5)
            emp_row[:] = np.nan
            emp.append(emp_row)
        else:
            emp.append(vec.flatten())
    emp = np.array(emp)
    emp = emp[1:-missing_thred,[2,3,4,0,1]]
    dis_mea = np.sqrt(np.sum(emp[:,[3,4]]**2,axis = 1)).reshape(-1,1)
    #since we don't track the z value, literally use 0 to alternate
    xyz = np.concatenate([emp[:,[3,4]],np.zeros(emp.shape[0]).reshape(-1,1)],axis = 1) 
    LLH_mea = convert_LLH(xyz,T)
    mea = np.concatenate([emp,LLH_mea,dis_mea],axis = 1)

    timestp = []
    for i in range(len(mea)):
        f = i + start_frame + 1
        timestp.append('%06.0f'%f)
    timestp = np.array(timestp).reshape(-1,1)
    objid = (np.ones(len(mea)) * key).astype(int).reshape(-1,1)
    summary = np.concatenate([objid,timestp,mea,est],axis = 1)
    summary = pd.DataFrame(summary,columns=col_info+col_mea+col_est)
    return summary

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

def process_traj_data(data): # alternate max length, return the data for classify
    data_test = data.loc[:,['ObjectID','Point_Cnt','Height','Length','Area']]
    data_temp_set = []
    for i,df in data_test.groupby('ObjectID'):
        df_temp = df.copy()
        # df_temp.Length = df_temp.Length.max()
        data_temp_set.append(df_temp)

    data_temp_set = pd.concat(data_temp_set)
    
    return data_temp_set

def classify_trajs(df,df_target,classifier):
    X_test = np.array(df_target.loc[:,['Point_Cnt','Height','Length','Area']]) 
    y_pred = classifier.predict(X_test)
    df.Length = df_target.Length
    df = pd.concat([df,pd.DataFrame(y_pred.reshape(-1,1),columns=['Class'])],axis = 1)
    return df 

def traj_post_processing(traj_df,length_thred,output_path):
    traj_post = []
    for i,t in traj_df.groupby('ObjectID'):
        if len(t) > length_thred:
            traj_post.append(t)
    traj_post = pd.concat(traj_post)

    traj_post.to_csv(output_path,index = False)

def get_thred(ts,d ,thred_s ,N ,delta_thred ,step):# Ransac Para
    ts_temp = ts.copy()
    ts_temp[ts_temp == 0] = 1000
    
    for i in range(N):
        sample = np.random.choice(ts_temp,replace=False)
        set_d = ts_temp[(ts_temp > sample - d)&(ts_temp < sample + d)]
        condition_thred = len(set_d)/len(ts_temp) > thred_s
        if condition_thred :
            break
            
    cur_thred = sample
    if i == (N -1):
        cur_thred = 1000


    while True:
        next_thred = cur_thred - step
        if (len(ts_temp[ts_temp > next_thred])/len(ts_temp) - len(ts_temp[ts_temp > cur_thred])/len(ts_temp)) < delta_thred:
            break
        cur_thred = next_thred

    return next_thred

def gen_bckmap(aggregated_maps , d, thred_s, N , delta_thred, step):
    thred_map = np.zeros((32,1800))
    for i in range(thred_map.shape[0]):
        for j in range(thred_map.shape[1]):
            thred_map[i,j] = get_thred(aggregated_maps[:,i,j],d,thred_s ,N, delta_thred ,step )
    return thred_map