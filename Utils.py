from os import stat
import numpy as np
from numpy.core.fromnumeric import shape
import open3d as op3
from scipy.optimize.optimize import main
from scipy.spatial import distance
import pandas as pd
from scipy.optimize import linear_sum_assignment
# import torch
from sklearn.cluster import DBSCAN

db_merge = DBSCAN(eps=3,min_samples=2)

# Kalman Filter Params

A = np.array( # x,y,l,w,h,x',y',l',w',h',x'',y''
    [[1,0,0,0,0,1,0,0,0,0,.5, 0],
     [0,1,0,0,0,0,1,0,0,0, 0,.5],
     [0,0,1,0,0,0,0,1,0,0, 0, 0],
     [0,0,0,1,0,0,0,0,1,0, 0, 0],
     [0,0,0,0,1,0,0,0,0,1, 0, 0],
     [0,0,0,0,0,1,0,0,0,0, 1, 0],
     [0,0,0,0,0,0,1,0,0,0, 0, 1],
     [0,0,0,0,0,0,0,1,0,0, 0, 0],
     [0,0,0,0,0,0,0,0,1,0, 0, 0],
     [0,0,0,0,0,0,0,0,0,1, 0, 0],
     [0,0,0,0,0,0,0,0,0,0, 1, 0],
     [0,0,0,0,0,0,0,0,0,0, 0, 1]]
      )
Q = np.diag([1,1,1,1,1,1,1,1,1,1,1,1])*0.01
H = np.array([[1,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0,0,0,0]])
R = np.diag([10,10,0.1,0.1,0.1])
P = np.diag([1,1,1,1,1,1,1,1,1,1,1,1])

class detected_obj():
    def __init__(self):
        self.glb_id = None
        self.start_frame = None
        self.missing_count = 0 # frame count of out of detection
        self.P = None
        self.state = None 
        self.label_seq = [] # represented labels at each frame 
        self.mea_seq = []
        self.post_seq = []
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
np.random.seed(150)
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

def extract_xylwh_merging_by_frame_interval(Labeling_map,Td_map,Thred_map,Background_map):
    
    XYZ,Labels = convert_point_cloud(Td_map,Labeling_map,Thred_map)
    unique_label = np.unique(Labels)
    if -1 in unique_label:
        unique_label = unique_label[1:]

    boundary_cols = []
    boundary_rows = []
    for l in unique_label:
        rows,cols = np.where(Labeling_map == l)
        sorted_cols_ind = np.argsort(cols)
        sorted_cols = cols[sorted_cols_ind]
        sorted_rows = rows[sorted_cols_ind]
        left_col,right_col = sorted_cols[0],sorted_cols[-1]
        
        if (right_col - left_col) >  900:
            left_col += 1800 
        boundary_cols.append([left_col,right_col])
        boundary_rows.append([rows[sorted_cols_ind[0]],rows[sorted_cols_ind[-1]]])
    boundary_cols,boundary_rows = np.array(boundary_cols),np.array(boundary_rows)

    sorted_label = np.argsort(boundary_cols[:,0])

    adjacent_label_pairs = []
    for sl in range(len(sorted_label) - 1):
        if boundary_cols[sorted_label[sl],1] < boundary_cols[sorted_label[sl+1],0]:
            adjacent_label_pairs.append([sorted_label[sl],sorted_label[sl+1]])
            
    if boundary_cols[sorted_label[-1],1] > 1800:
        if (boundary_cols[sorted_label[-1],1] - 1800) < boundary_cols[sorted_label[0],0]:
            adjacent_label_pairs.append([sorted_label[-1],sorted_label[0]])
    else:
        if boundary_cols[sorted_label[-1],1] < boundary_cols[sorted_label[0],0]:
            adjacent_label_pairs.append([sorted_label[-1],sorted_label[0]])
    Merge_cobs = []
    for adjacent in adjacent_label_pairs:
        pair_a,pair_b = adjacent[0],adjacent[1]
        interval_left_col,interval_right_col = boundary_cols[pair_a][1],boundary_cols[pair_b][0]
        interval_left_row,interval_right_row = boundary_rows[pair_a][1],boundary_rows[pair_b][0]
    #     print(interval_left_col,interval_right_col)
        if (interval_right_col - interval_left_col) > 30:
            continue
        high = interval_left_row
        low = interval_right_row
        if high < low: 
            high,low = low,high
        
        interval_map = Td_map[low:high+1,interval_left_col:interval_right_col+1][Background_map[low:high+1,interval_left_col:interval_right_col+1]]
        if len(interval_map) == 0 :
            continue
        min_dis_int = interval_map.min()
        min_dis_a = Td_map[Labeling_map == pair_a].min()
        min_dis_b = Td_map[Labeling_map == pair_b].min()
        if (min_dis_int + 1.2 < min_dis_a)&(min_dis_int + 1.2 <min_dis_b)&(np.abs(min_dis_a - min_dis_b) < 1.2):
            Merge_cobs.append([pair_a,pair_b])
    

    for cob in Merge_cobs:
        for i in range(1,len(cob)):
            Labeling_map[Labeling_map == cob[i]] = cob[0]
            unique_label[unique_label == cob[i]] = cob[0]
            Labels[Labels == cob[i]] =cob[0]
            
    new_uni_labels = np.unique(unique_label)
    # new_uni_labels_ = np.arange(len(new_uni_labels))
    # XYZ,Labels = convert_point_cloud(Td_map,Labeling_map,Thred_map)
    # for i,n_l in enumerate(new_uni_labels_):
    #     Labeling_map[Labeling_map == new_uni_labels[i]] = n_l
        
    xylwh_set = []  
    for l in new_uni_labels:
        point = XYZ[Labels == l]
        xylwh = get_params_from_detection_points(point)
        xylwh_set.append(xylwh)

    return np.array(xylwh_set),new_uni_labels,Labeling_map


def extract_xylwh_merging_by_frame_db(Labeling_map,Td_map,Thred_map):
    
    XYZ,Labels = convert_point_cloud(Td_map,Labeling_map,Thred_map)
    uni_labels = np.unique(Labels)
    if -1 in uni_labels:
        uni_labels = uni_labels[1:]
    Point_sets = []
    Center_points = []
    for l in uni_labels:
        points = XYZ[Labels == uni_labels[l]][:,:2]
        center_point = np.mean(points,axis = 0)
        Center_points.append(center_point)
        Point_sets.append(points)
    Center_points = np.array(Center_points)
    Merge_labels = db_merge.fit_predict(Center_points)
    uni_mer_labels = np.unique(Merge_labels)
    if -1 in uni_mer_labels:
        uni_mer_labels = uni_mer_labels[1:]
    Merge_cobs = []
    for l in uni_mer_labels:
        Merge_cobs.append(uni_labels[Merge_labels == l])
    for cob in Merge_cobs:
        for i in range(1,len(cob)):
            Labeling_map[Labeling_map == cob[i]] = cob[0]
            uni_labels[uni_labels == cob[i]] = cob[0]
            Labels[Labels == cob[i]] =cob[0]
            
    new_uni_labels = np.unique(uni_labels)
    # new_uni_labels_ = np.arange(len(new_uni_labels))
    # XYZ,Labels = convert_point_cloud(Td_map,Labeling_map,Thred_map)
    # for i,n_l in enumerate(new_uni_labels_):
    #     Labeling_map[Labeling_map == new_uni_labels[i]] = n_l
        
    xylwh_set = []  
    for l in new_uni_labels:
        point = XYZ[Labels == l]
        xylwh = get_params_from_detection_points(point)
        xylwh_set.append(xylwh)

    return np.array(xylwh_set),new_uni_labels,Labeling_map


def extract_xylwh_by_frame(Labeling_map,Td_map,Thred_map):
    
    XYZ,Labels = convert_point_cloud(Td_map,Labeling_map,Thred_map)
    if -1 in Labels:
        unique_id = np.unique(Labels)[1:]
    else:
        unique_id = np.unique(Labels)
    xylwh_set = []  
    for l in unique_id:
        xylwh = get_params_from_detection_points(Td_map,Labeling_map,l,Thred_map)
        xylwh_set.append(xylwh)
    
    return np.array(xylwh_set),unique_id

def extract_mea_state_vec(xylwh_set):
    return xylwh_set.reshape((-1,xylwh_set.shape[1],1))

def create_new_detection_NN(Tracking_pool,Global_id,state_init,label_init,mea_init,start_frame):
    
    new_detection = detected_obj()
    new_detection.glb_id = Global_id
    new_detection.state = state_init
    new_detection.label_seq.append(label_init)
    new_detection.start_frame = start_frame
    new_detection.mea_seq.append(mea_init)
    new_detection.post_seq.append(state_init)
    Tracking_pool[Global_id] = new_detection

def create_new_detection(Tracking_pool,Global_id,P_init,state_init,label_init,mea_init,start_frame):
    
    new_detection = detected_obj()
    new_detection.glb_id = Global_id
    new_detection.P = P_init
    new_detection.state = state_init
    new_detection.label_seq.append(label_init)
    new_detection.start_frame = start_frame
    new_detection.mea_seq.append(mea_init)
    new_detection.post_seq.append(state_init)
    Tracking_pool[Global_id] = new_detection
    
def process_fails(Tracking_pool,Off_tracking_pool,glb_id,state_cur_,P_cur_,missing_thred):
    if Tracking_pool[glb_id].missing_count > missing_thred:
        Off_tracking_pool[glb_id] = Tracking_pool.pop(glb_id)
    else:
        Tracking_pool[glb_id].missing_count += 1
        Tracking_pool[glb_id].state = state_cur_
        Tracking_pool[glb_id].P = P_cur_
        Tracking_pool[glb_id].label_seq.append(-1)
        Tracking_pool[glb_id].mea_seq.append(-1)
        Tracking_pool[glb_id].post_seq.append(state_cur_)
        
def process_fails_NN(Tracking_pool,Off_tracking_pool,glb_id,state_cur_,missing_thred):
    if Tracking_pool[glb_id].missing_count > missing_thred:
        Off_tracking_pool[glb_id] = Tracking_pool.pop(glb_id)
    else:
        Tracking_pool[glb_id].missing_count += 1
        Tracking_pool[glb_id].state = state_cur_
        Tracking_pool[glb_id].label_seq.append(-1)
        Tracking_pool[glb_id].mea_seq.append(-1)
        Tracking_pool[glb_id].post_seq.append(state_cur_)

def associate_detections_NN(Tracking_pool,glb_id,state,next_label,mea_next):
    
    Tracking_pool[glb_id].state = state
    Tracking_pool[glb_id].label_seq.append(next_label)
    Tracking_pool[glb_id].mea_seq.append(mea_next)
    Tracking_pool[glb_id].post_seq.append(state)
    Tracking_pool[glb_id].missing_count = 0

def associate_detections(Tracking_pool,glb_id,state,P,next_label,mea_next):
    
    Tracking_pool[glb_id].state = state
    Tracking_pool[glb_id].P = P
    Tracking_pool[glb_id].label_seq.append(next_label)
    Tracking_pool[glb_id].mea_seq.append(mea_next)
    Tracking_pool[glb_id].post_seq.append(state)
    Tracking_pool[glb_id].missing_count = 0

def state_predict_NN(state,tracking_nn):
    
    temp = state.copy()
    cur_xy = state[:,[0,1]].copy()
    torch_state = torch.tensor(temp[:,[0,1,-2,-1]]).float()
    torch_state[:,[-2,-1]] = torch_state[:,[-2,-1]] / 10 # convert t0 m/0.1s 
    pred_xy = tracking_nn(torch_state).detach().numpy() # n x 2
    temp[:,[0,1]] = pred_xy
    speed_xy = pred_xy - cur_xy
    temp[:,[-2,-1]] = speed_xy
    
    return  temp

def state_predict(A,Q,state,P):
    """
    state: s_k-1, (n x 10 x 1)
    Cov: P_k-1 (n x 10 x 10)
    """
    state_ = np.matmul(A,state)
    P_ = np.matmul(np.matmul(A,P),A.transpose()) + Q
    return state_,P_

def state_update_NN(state_,mea):
    
    state_[:,:2] = (state_[:,:2] + mea[:,:2])/2
    return state_
    
def state_update(A,H,state_,P_,R,mea):
    """
    mea: m_k (m x 5 x 1)
    
    """
    K = np.matmul(np.matmul(P_,H.transpose()),np.linalg.inv(np.matmul(np.matmul(H,P_),H.transpose()) + R))
    P = np.matmul((np.eye(A.shape[0]) - np.matmul(K,H)),P_)
    residual = mea - np.matmul(H,state_) # n x 5 x 1
    state = state_ + np.matmul(K,residual)
    
    return state, P 


def get_affinity_mat_cos(state,state_,P_,mea):
    State_affinity = np.zeros((state_.shape[0],mea.shape[0]))
    for i,s_ in enumerate(state_):
        v_ = s_.copy().flatten()
        v = state[i].copy().flatten()
        # VI = P_[i][:2,:2].copy()
        v_all = v_[:2]
        
        for j,m in enumerate(mea):
            u = m.copy().flatten()
            d_cur_mea = np.sqrt(np.sum((v[:2] - u[:2])**2))
            if d_cur_mea > 5:
                simi = 1e4
            else:
                d_pred_mea = np.sqrt(np.sum((v_all - u[:2])**2))
                v_cur = np.sqrt(np.sum(v[5:7]**2))
                pred_vec = v_[5:7]
                mea_vec = u[:2] - v[:2]
                norm_pred_vec,norm_mea_vec = np.sqrt(np.sum(pred_vec**2)),np.sqrt(np.sum(mea_vec**2))
                if (norm_mea_vec == 0) | (norm_pred_vec == 0):
                    cos_mea_pred = 0
                else:
                    cos_mea_pred = (mea_vec * pred_vec).sum()/(norm_pred_vec*norm_mea_vec)
                simi = d_pred_mea/((-np.e**(-v_cur)+2) * (cos_mea_pred + 1))
            # simi_embed = distance.mahalanobis(u[2:],v_embed,VI_embed)
            State_affinity[i][j] = simi

            
    return State_affinity


def get_affinity_mat(state,state_,P_,mea):
    State_affinity = np.zeros((state_.shape[0],mea.shape[0]))
    for i,s_ in enumerate(state_):
        v_ = s_.copy().flatten()
        v = state[i].copy().flatten()
        # VI = P_[i][:2,:2].copy()
        v_all = v_[:2]

        
        for j,m in enumerate(mea):
            u = m.copy().flatten()
            # d = np.sqrt(np.sum((v[:2] - u[:2])**2))
            simi = np.sqrt(np.sum((v_all[:2] - u[:2])**2))
            # simi_ np.sqrt(np.sum((v_all[:2] - u[:2])**2))
            # simi_embed = distance.mahalanobis(u[2:],v_embed,VI_embed)
            State_affinity[i][j] = simi

            
    return State_affinity

def get_affinity_mat_NN(state_cur_,mea_next):
    
    State_affinity_0 = np.empty((state_cur_.shape[0],mea_next.shape[0]))
    # State_affinity_1 = np.empty((state_cur_.shape[0],mea_next.shape[0])) 
    State_affinity_0.fill(1e3)
    # State_affinity_1.fill(np.inf)
    
    for i,s_ in enumerate(state_cur_):    
        v_ = s_.copy()
        for j,m in enumerate(mea_next):
            u = m.copy()
            d = np.sum((v_[:2] - u[:2])**2) #Distance match
            if d < 49 :
                State_affinity_0[i][j] = d
            
    return State_affinity_0

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


col_info =['ObjectID','FrameIndex']
col_est = ['Object_Length_est','Object_Width_est','Object_Height_est','Coord_X_est','Coord_Y_est','Coord_Lon_est','Coord_Lat_est','Coord_Evel_est','Coord_Dis_est','Speed_x','Speed_y','Speed_est']
col_mea = ['Object_Length_mea','Object_Width_mea','Object_Height_mea','Coord_X_mea','Coord_Y_mea','Coord_Lon_mea','Coord_Lat_mea','Coord_Evel_mea','Coord_dis_mea']

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


