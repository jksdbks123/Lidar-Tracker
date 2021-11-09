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

# # Kalman Filter Params

# # A = np.array( # x,y,l,w,h,x',y',l',w',h',x'',y''
# #     [[1,0,0,0,0,1,0,0,0,0,.5, 0],
# #      [0,1,0,0,0,0,1,0,0,0, 0,.5],
# #      [0,0,1,0,0,0,0,1,0,0, 0, 0],
# #      [0,0,0,1,0,0,0,0,1,0, 0, 0],
# #      [0,0,0,0,1,0,0,0,0,1, 0, 0],
# #      [0,0,0,0,0,1,0,0,0,0, 1, 0],
# #      [0,0,0,0,0,0,1,0,0,0, 0, 1],
# #      [0,0,0,0,0,0,0,1,0,0, 0, 0],
# #      [0,0,0,0,0,0,0,0,1,0, 0, 0],
# #      [0,0,0,0,0,0,0,0,0,1, 0, 0],
# #      [0,0,0,0,0,0,0,0,0,0, 1, 0],
# #      [0,0,0,0,0,0,0,0,0,0, 0, 1]]
# #       )
# # Q = np.diag([1,1,1,1,1,0.1,0.1,1,1,1,0.01,0.01])*0.01
# # H = np.array([[1,0,0,0,0,0,0,0,0,0,0,0],
# #             [0,1,0,0,0,0,0,0,0,0,0,0],
# #             [0,0,1,0,0,0,0,0,0,0,0,0],
# #             [0,0,0,1,0,0,0,0,0,0,0,0],
# #             [0,0,0,0,1,0,0,0,0,0,0,0]])
# # P = np.diag([1,1,1,1,1,1,1,1,1,1,1,1])*100

A = np.array([ # x,y,x',y',x'',y''
    [1,0,1,0,.5,0],
    [0,1,0,1,0,.5],
    [0,0,1,0,1,0],
    [0,0,0,1,0,1],
    [0,0,0,0,1,0],
    [0,0,0,0,0,1]
])
Q = np.diag([1,1,0.1,0.1,0.001,0.001])
R = np.diag([0.01,0.01])
P = np.diag([0.1,0.1,0.3,0.3,0.01,0.01])

H = np.array([
    [1,0,0,0,0,0],
    [0,1,0,0,0,0]
])




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


def extract_xy_interval_merging_TR(Labeling_map,Td_map,Background_map,thred_merge = 25):
        
    
    unique_label = np.unique(Labeling_map)
    #Only Background contains 
    if -1 in unique_label:
        unique_label = unique_label[1:]
        
    if len(unique_label) < 1:
        return np.array([]),unique_label,Labeling_map
    occlusion_indicator = -np.ones((len(azimuths))).astype('int')
    rowses = []
    colses = []
    for l in unique_label:
        rows,cols = np.where(Labeling_map == l)
        occlusion_indicator[cols] = l
        rowses.append(rows)
        colses.append(cols)
    TSAv = occlusion_indicator != -1
    counts,appears = count(TSAv)
    Merge_cobs = []
    ind_pairs = [(i,i+1) for i in range(len(counts) - 1)]
    ind_pairs += [(-1,0)]
    for pair in ind_pairs:
        col_right = appears[pair[0]] + counts[pair[0]] - 1
        col_left = appears[pair[1]]
        is_normal = col_left >= col_right
        if is_normal&((col_left - col_right) > thred_merge):
            continue
        bounder_right_label = occlusion_indicator[col_right]
        bounder_left_label = occlusion_indicator[col_left]
        # right-bound on left ---- left-boundon right
        label_ind_right= np.where(unique_label == bounder_right_label)[0][0]
        rows_right = rowses[label_ind_right][colses[label_ind_right] == col_right]
        label_ind_left = np.where(unique_label == bounder_left_label)[0][0]
        rows_left = rowses[label_ind_left][colses[label_ind_left] == col_left]
        rows_2bounds = np.concatenate([rows_left,rows_right])
        high,low = rows_2bounds.max(),rows_2bounds.min()
        if is_normal:
            interval_map = Td_map[low:high+1,col_right:col_left+1][Background_map[low:high+1,col_right:col_left+1]]
        else:
            if (len(azimuths) - col_right) + col_left > thred_merge:
                continue
            else: 
                interval_map_right = Td_map[low:high+1,col_right:len(azimuths)][Background_map[low:high+1,col_right:len(azimuths)]]
                interval_map_left = Td_map[low:high+1,:col_left + 1][Background_map[low:high+1,:col_left + 1]]
                interval_map = np.concatenate([interval_map_left,interval_map_right])

        if len(interval_map) == 0 :
            continue
        min_dis_int = interval_map.min()
        min_dis_right = Td_map[rows_right,col_right].min()
        min_dis_left = Td_map[rows_left,col_left].min()
        if (min_dis_int  < min_dis_right)&(min_dis_int < min_dis_left)&(np.abs(min_dis_right - min_dis_left) < 2.5):
            Merge_cobs.append([label_ind_right,label_ind_left])

    for cob in Merge_cobs:
        for i in range(1,len(cob)):
            Labeling_map[Labeling_map == cob[i]] = cob[0]
            unique_label[unique_label == cob[i]] = cob[0]

    new_uni_labels = np.unique(unique_label)
    xy_set = []
    for label in new_uni_labels:
        rows,cols = np.where(Labeling_map == label)
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
        xy_set.append(get_representative_point(refer_rows,refer_cols,Td_map))
    
    return np.array(xy_set),new_uni_labels,Labeling_map


def get_representative_point(ref_rows,ref_cols,Td_map): 
    td_freq_map = Td_map
    longitudes = theta[ref_rows]*np.pi / 180
    latitudes = azimuths[ref_cols] * np.pi / 180 
    hypotenuses = td_freq_map[ref_rows,ref_cols] * np.cos(longitudes)
    X = hypotenuses * np.sin(latitudes)
    Y = hypotenuses * np.cos(latitudes)
    Z = td_freq_map[ref_rows,ref_cols] * np.sin(longitudes)
    
    return np.array([
        [X[0],Y[0]],
        [X[1],Y[1]]
    ]).reshape(2,2,1) # n_repr x xy_dim x 1 

def linear_assignment_modified(State_affinity):
    State_affinity_temp = State_affinity.copy()
    associated_ind_glb,associated_ind_label = [],[]
    for i,p in enumerate(State_affinity):
        if (p != 0).sum() == 1:
            associated_ind_glb.append(i)
            label_ind = np.where(p != 0)[0][0]
            associated_ind_label.append(label_ind)
            State_affinity_temp[i,label_ind] = 0
    
    associated_ind_glb_extend_,associated_ind_labels_extend_= linear_sum_assignment(State_affinity_temp,maximize=True)
    
    for i in range(len(associated_ind_glb_extend_)):
        if State_affinity_temp[associated_ind_glb_extend_[i],associated_ind_labels_extend_[i]] != 0:
            associated_ind_glb.append(associated_ind_glb_extend_[i])
            associated_ind_label.append(associated_ind_labels_extend_[i])
            
    associated_ind_glb,associated_ind_label = np.array(associated_ind_glb),np.array(associated_ind_label)
    ind = np.argsort(associated_ind_glb)
    associated_ind_glb = associated_ind_glb[ind]
    associated_ind_label = associated_ind_label[ind]
    
    return associated_ind_glb,associated_ind_label
    

def linear_assignment_modified_dis(State_affinity,thred = 9):

    State_affinity_temp = State_affinity.copy()
    associated_ind_glb,associated_ind_label = [],[]
    for i,dis in enumerate(State_affinity):
        if (dis < thred).sum() == 1:
            associated_ind_glb.append(i)
            label_ind = np.where(dis < thred)[0][0]
            associated_ind_label.append(label_ind)
            State_affinity_temp[i,label_ind] = 1e3
    associated_ind_glb_extend_,associated_ind_labels_extend_= linear_sum_assignment(State_affinity_temp,maximize = False)
    for i in range(len(associated_ind_glb_extend_)):
        if State_affinity_temp[associated_ind_glb_extend_[i],associated_ind_labels_extend_[i]] < thred:
            associated_ind_glb.append(associated_ind_glb_extend_[i])
            associated_ind_label.append(associated_ind_labels_extend_[i])
    associated_ind_glb,associated_ind_label = np.array(associated_ind_glb),np.array(associated_ind_label)
    ind = np.argsort(associated_ind_glb)
    associated_ind_glb = associated_ind_glb[ind]
    associated_ind_label = associated_ind_label[ind]

    return associated_ind_glb,associated_ind_label

def linear_assignment_modified_jpd(State_affinity):
    State_affinity_temp = State_affinity.copy()
    associated_ind_glb,associated_ind_label = [],[]
    for i,p in enumerate(State_affinity):
        if (p != 0).sum() == 1:
            associated_ind_glb.append(i)
            label_ind = np.where(p != 0)[0][0]
            associated_ind_label.append(label_ind)
            State_affinity_temp[i,label_ind] = 0
    
    associated_ind_glb_extend_,associated_ind_labels_extend_= linear_sum_assignment(State_affinity_temp,maximize=True)
    
    for i in range(len(associated_ind_glb_extend_)):
        if State_affinity_temp[associated_ind_glb_extend_[i],associated_ind_labels_extend_[i]] != 0:
            associated_ind_glb.append(associated_ind_glb_extend_[i])
            associated_ind_label.append(associated_ind_labels_extend_[i])
            
    associated_ind_glb,associated_ind_label = np.array(associated_ind_glb),np.array(associated_ind_label)
    ind = np.argsort(associated_ind_glb)
    associated_ind_glb = associated_ind_glb[ind]
    associated_ind_label = associated_ind_label[ind]
    
    return associated_ind_glb,associated_ind_label


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
    
    if np.sqrt(np.sum(state_init[0][:2]**2)) > 20:
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
    fail_condition1 = Tracking_pool[glb_id].missing_count > missing_thred
    fail_condition2 = np.sqrt(np.sum(state_cur_[0][:2]**2)) > 75
    if fail_condition1|fail_condition2:
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
from scipy.stats import multivariate_normal

def get_affinity_mat_jpd(state,state_,P_,mea):
    State_affinity = np.zeros((state_.shape[0],mea.shape[0]))
    for i,s_ in enumerate(state_):
        v_ = s_.copy().flatten()
        v_cur = state[i].flatten()
        a = [0,1]
        v_all = v_[a]
        cov = np.zeros((len(a),len(a)))
        for i_emp,i_P in enumerate(a):
            for j_emp,j_P in enumerate(a):
                cov[i_emp,j_emp] = P_[i][i_P,j_P]

        var = multivariate_normal(mean=v_all, cov=cov)
        for j,m in enumerate(mea):
            u = m.copy().flatten()
            
            v_next = np.sqrt(np.sum((v_cur[:2] - u[:2])**2))
            # v_previous = np.sqrt(np.sum(v_cur[5:7]**2))
            if (v_next > 5):
                State_affinity[i][j] = 0
            else:
                jp = var.pdf(u[a])
                State_affinity[i][j] = jp

    return State_affinity

def get_affinity_mat_jpd_TR(state,state_,P_,mea):
    State_affinity = np.zeros((state_.shape[1],state_.shape[0],mea.shape[0]))
    for i,s_ in enumerate(state_):
         # includes the pred states for two reprs 
         # s_: 2 x 6 x 1
        state_cur = state[i].copy().reshape(2,-1)[:,:2]
        state_pred = s_.copy().reshape(2,-1)[:,:2]
        
         # cov_tr : 2 x 6 x 6 
        cov_tr = P_[i][:,:2,:2]
        var_tr = [multivariate_normal(mean=state_pred[k], cov=cov_tr[k]) for k in range(state_cur.shape[0])]
        for j,m in enumerate(mea):
            mea_next = m.copy().reshape(2,-1)
            for k in range(s_.shape[0]):
                dis_error = np.sqrt(np.sum((state_pred[k] - mea_next[k])**2))
                if dis_error < 7:
                    jp = var_tr[k].pdf(mea_next[k])
                    State_affinity[k,i,j] = jp

    return np.max(State_affinity,axis = 0)

def cal_heading_vec(post_seq):

    post_seq = np.array(post_seq)
    if len(post_seq) >= 5:
        heading_vec = np.sum(post_seq[-5:,:,2:4],axis = 0)
    else:
        heading_vec = np.sum(post_seq[:,:,2:4],axis = 0)
    return heading_vec # 2 x 2 x 1 

def get_affinity_mat_mal_heading_TR(state_cur,heading_vecs,state_,P_,mea,heading_step = 5):
    State_affinity = 1e3*np.ones((state_.shape[1],state_.shape[0],mea.shape[0]))
    temp_state = state_cur.copy()
    for i,s_ in enumerate(state_):
         # includes the pred states for two reprs 
         # s_: 2 x 6 x 1
         # cov_tr : 2 x 6 x 6 
        
        speed_cur = temp_state[i].copy().reshape(2,-1)[:,2:4]
        # 2 x 2
        state_cur = temp_state[i].copy().reshape(2,-1)[:,:2]
        # 2 x 2
        
        speed_cur = np.sqrt(np.sum(speed_cur**2,axis = 1))
        # 1 x 2
        heading_vec = heading_vecs[i].copy().reshape(2,-1)
        # 2 x 2
        state_pred = s_.copy().reshape(2,-1)[:,:2]
        # 2 x 2
        cov_tr = P_[i][:,:2,:2]
        
        for j,m in enumerate(mea):
            mea_next = m.copy().reshape(2,-1)
            # 2 x 2
            for k in range(s_.shape[0]):
                mal_dis = distance.mahalanobis(mea_next[k],state_pred[k],np.linalg.inv(cov_tr[k]))
                vec_mea = mea_next[k] - state_cur[k] # vec between mea and cur est
                # 1 x 2
                vector_1 = heading_vec[k]
                if (vector_1 == 0).all():
                    cos_angle = 0
                else:
                # 1 x 2
                    vector_2 = vec_mea
                    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
                    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
                    cos_angle = np.dot(unit_vector_2 , unit_vector_1)
                    # cos of angle: 1 x 2 

                # cos of the angle between mea_vec and heading 
                # ranged from -1 ~ 1 
                if (mal_dis < 3):
                    speed_next = np.sqrt(np.sum((mea_next[k] - state_cur[k])**2))
                    speed_diff = np.abs(speed_next - speed_cur[k]) 
                    if cos_angle<0.7:
                        cos_angle = 0
                    if speed_diff > 0.1:
                        speed_diff = 3
                    State_affinity[k,i,j] = mal_dis + 3*(1-cos_angle) + speed_diff                      

    return np.min(State_affinity,axis = 0)

def get_affinity_mat_mal_TR(state,state_,P_,mea):
    State_affinity = 1e3*np.ones((state_.shape[1],state_.shape[0],mea.shape[0]))
    for i,s_ in enumerate(state_):
         # includes the pred states for two reprs 
         # s_: 2 x 6 x 1
        # state_cur = state[i].copy().reshape(2,-1)[:,:2]
        state_pred = s_.copy().reshape(2,-1)[:,:2]
         # cov_tr : 2 x 6 x 6 
        cov_tr = P_[i][:,:2,:2]
        for j,m in enumerate(mea):
            mea_next = m.copy().reshape(2,-1)
            for k in range(s_.shape[0]):
                mal_dis = distance.mahalanobis(mea_next[k],state_pred[k],np.linalg.inv(cov_tr[k]))
                if mal_dis < 3:
                    State_affinity[k,i,j] = mal_dis
    return np.min(State_affinity,axis = 0)

def get_affinity_mat_dis_TR(state,state_,P_,mea):
    State_affinity = 1e3*np.ones((state_.shape[1],state_.shape[0],mea.shape[0]))
    for i,s_ in enumerate(state_):
         # includes the pred states for two reprs 
         # s_: 2 x 6 x 1
        state_cur = state[i].copy().reshape(2,-1)[:,:2]
        state_pred = s_.copy().reshape(2,-1)[:,:2]
        for j,m in enumerate(mea):
            mea_next = m.copy().reshape(2,-1)
            for k in range(s_.shape[0]):
                dis_error = np.sqrt(np.sum((state_pred[k] - mea_next[k])**2))
                if dis_error < 2:
                    State_affinity[k,i,j] = dis_error

    return np.min(State_affinity,axis = 0)

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
column_names_TR = ['ObjectID','FrameIndex','Coord_X_Mea','Coord_Y_Mea','Coord_Z_Mea','Distance_Mea','Longitude_Mea','Latitude_Mea',
                'Elevation_Mea','Coord_X_Est','Coord_Y_Est','Coord_Z_Est','Distance_Est','Speed_X','Speed_Y','Speed(m/s)','Acc_X','Acc_Y','Longitude_Est','Latitude_Est','Elevation_Est']

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

def get_summary_file_TR(post_seq,mea_seq,key,start_frame,missing_thred,T):
    temp = np.array(post_seq)
    temp = temp.reshape((temp.shape[0],temp.shape[1],temp.shape[2]))
    # n x 2 x 6
    temp_xy = temp[:,:,:2]
    # n x 2 x 2
    dis_est = np.sqrt((temp_xy[:,:,0]**2 + temp_xy[:,:,1]**2))
    # n x 2 
    speed_xy = temp[:,:,2:4] * 10 
    # n x 2 x 2
    speed = np.sqrt((speed_xy[:,:,0]**2 + speed_xy[:,:,1]**2))*3600/1000
    # n x 2
    acc_xy = temp[:,:,4:6]
    # n x 2
    xyz_0 = np.concatenate([temp_xy[:,0],np.zeros(len(temp_xy)).reshape(-1,1)],axis = 1)
    xyz_1 = np.concatenate([temp_xy[:,1],np.zeros(len(temp_xy)).reshape(-1,1)],axis = 1)
    LLH_est_0 = convert_LLH(xyz_0,T)
    LLH_est_1 = convert_LLH(xyz_1,T)
    est_0 = np.concatenate([xyz_0,dis_est[:,0].reshape(-1,1),speed_xy[:,0],speed[:,0].reshape(-1,1),acc_xy[:,0],LLH_est_0],axis = 1)
    est_1 = np.concatenate([xyz_1,dis_est[:,1].reshape(-1,1),speed_xy[:,1],speed[:,1].reshape(-1,1),acc_xy[:,1],LLH_est_1],axis = 1)
    temp = mea_seq
    emp_0,emp_1 = [],[]
    for i,vec in enumerate(temp):
        if type(vec) == int:
            emp_row = np.empty(2)
            emp_row[:] = np.nan
            emp_0.append(emp_row)
            emp_1.append(emp_row)
        else:
            emp_0.append(vec[0].flatten())
            emp_1.append(vec[1].flatten())
    emp_0,emp_1 = np.array(emp_0),np.array(emp_1)
    dis_mea_0,dis_mea_1 = np.sqrt(np.sum(emp_0**2,axis = 1)).reshape(-1,1),np.sqrt(np.sum(emp_1**2,axis = 1)).reshape(-1,1)
    xyz_0 = np.concatenate([emp_0,np.zeros(len(emp_0)).reshape(-1,1)],axis  = 1)
    xyz_1 = np.concatenate([emp_1,np.zeros(len(emp_1)).reshape(-1,1)],axis  = 1)
    LLH_est_0 = convert_LLH(xyz_0,T)
    LLH_est_1 = convert_LLH(xyz_1,T)
    mea_0 = np.concatenate([xyz_0,dis_mea_0,LLH_est_0],axis = 1)
    mea_1 = np.concatenate([xyz_1,dis_mea_1,LLH_est_1],axis = 1)
    timestp = []
    for i in range(len(temp)):
        f = i + start_frame + 1
        timestp.append('%06.0f'%f)
    timestp = np.array(timestp).reshape(-1,1)
    objid = (np.ones(len(temp)) * key).astype(int).reshape(-1,1)
    summary_0 = np.concatenate([objid,timestp,mea_0,est_0],axis = 1)
    summary_1 = np.concatenate([objid,timestp,mea_1,est_1],axis = 1)
    
    summary_0 = pd.DataFrame(summary_0,columns=column_names_TR)
    summary_1 = pd.DataFrame(summary_1,columns=column_names_TR)
    return summary_0,summary_1

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


