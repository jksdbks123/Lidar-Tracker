
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
import socket

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
        self.P_seq = []
        self.P = None
        self.missing_count = 0
        

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
A = np.array([ # x,y,x',y'
    [1,0,1,0],
    [0,1,0,1],
    [0,0,1,0],
    [0,0,0,1],
])
Q = np.diag([1,1,0.1,0.1])
R = np.diag([1,1])
P_em = np.diag([1.53,1.53,2.2,2.2])

H = np.array([
    [1,0,0,0],
    [0,1,0,0]
])
db_merge = DBSCAN(eps = 1.8, min_samples = 2)
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

def state_predict(A,Q,state,P):
    """
    state: s_k-1, (n x 10 x 1)
    Cov: P_k-1 (n x 10 x 10)
    """
    state_ = np.matmul(A,state)
    P_ = np.matmul(np.matmul(A,P),A.transpose()) + Q
    return state_,P_

def state_update(A,H,state_,P_,R,mea):
    """
    mea: m_k (m x 5 x 1)
    
    """
    K = np.matmul(np.matmul(P_,H.transpose()),np.linalg.inv(np.matmul(np.matmul(H,P_),H.transpose()) + R))
    P = np.matmul((np.eye(A.shape[0]) - np.matmul(K,H)),P_)
    residual = mea - np.matmul(H,state_) # n x 5 x 1
    state = state_ + np.matmul(K,residual)
    
    return state, P 

def get_params_from_detection_points(point):
    
    pcd = op3.geometry.PointCloud()
    pcd.points = op3.utility.Vector3dVector(point)
    bbox = pcd.get_axis_aligned_bounding_box()
    # x,y,length,width,height 
    xylwh = np.concatenate([bbox.get_center()[:2],bbox.get_max_bound() - bbox.get_min_bound()])
    return xylwh


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

def get_xy_set(new_uni_labels,Labeling_map,Td_map,if_app):
    xy_set = [] # xy position and apperance features
    if if_app:
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
        if if_app:
            apperance = get_appearance_features(rows_temp,cols_temp,Td_map)
            apperance_set.append(apperance)
        xy = get_representative_point(refer_rows,refer_cols,Td_map) # x,y vec for two representatives 
        xy_set.append(xy)
    xy_set = np.array(xy_set)
    if if_app:
        apperance_set = np.array(apperance_set)
        return xy_set,apperance_set
    else:
        return xy_set

def extract_xy(Labeling_map,Td_map):
        
    # Plane_model is a 1 x 4 array representing a,b,c,d in ax + by + cz + d = 0 
    new_uni_labels = np.unique(Labeling_map[Labeling_map != -1])

    xy_set = get_xy_set(new_uni_labels,Labeling_map,Td_map,False)
    if len(xy_set) > 0:
        total_labels = np.concatenate([new_uni_labels,new_uni_labels])
        edge_points = np.concatenate([xy_set[:,1,:,0],xy_set[:,0,:,0]])
        merge_labels = db_merge.fit_predict(edge_points)
        unique_merge_labels = np.unique(merge_labels[merge_labels != -1])
        merge_pairs = [total_labels[merge_labels == l] for l in unique_merge_labels]
        for p in merge_pairs:
            merging_p = np.unique(p)
            if len(merging_p) > 1:
                for i in range(1,len(merging_p)):
                    Labeling_map[Labeling_map == merging_p[i]] = merging_p[0]
        new_uni_labels = np.unique(Labeling_map[Labeling_map != -1])
        xy_set,apperance_set = get_xy_set(new_uni_labels,Labeling_map,Td_map,True)
        return xy_set,apperance_set,new_uni_labels,Labeling_map
    else:
        return xy_set,[],new_uni_labels,Labeling_map



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
    area = b1 * b2
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
    # ind = np.argsort(associated_ind_cur)
    # associated_ind_cur = associated_ind_cur[ind]
    # associated_ind_next = associated_ind_next[ind]

    return associated_ind_cur,associated_ind_next

def linear_assignment_kalman(State_affinity):

    associated_ind_cur,associated_ind_next = [],[]
    associated_ind_cur_extend_,associated_ind_next_extend_= linear_sum_assignment(State_affinity,maximize = False)
    for i in range(len(associated_ind_cur_extend_)):
        if State_affinity[associated_ind_cur_extend_[i],associated_ind_next_extend_[i]] < 1.5:
            associated_ind_cur.append(associated_ind_cur_extend_[i])
            associated_ind_next.append(associated_ind_next_extend_[i])
    associated_ind_cur,associated_ind_next = np.array(associated_ind_cur),np.array(associated_ind_next)
    # ind = np.argsort(associated_ind_cur)
    # associated_ind_cur = associated_ind_cur[ind]
    # associated_ind_next = associated_ind_next[ind]

    return associated_ind_cur,associated_ind_next



def create_new_detection(Tracking_pool,Global_id,state_init,app_init,label_init,mea_init,P,start_frame):
    new_detection = detected_obj()
    new_detection.glb_id = Global_id
    new_detection.state = state_init
    new_detection.apperance = app_init
    new_detection.P = P
    new_detection.label_seq.append(label_init)
    new_detection.start_frame = start_frame
    new_detection.mea_seq.append(mea_init)
    new_detection.post_seq.append(state_init)
    new_detection.app_seq.append(app_init)
    new_detection.P_seq.append(P)
    Tracking_pool[Global_id] = new_detection
        


def process_fails(Tracking_pool,Off_tracking_pool,glb_id,state_cur_,P_cur_,missing_thred):
    Tracking_pool[glb_id].missing_count += 1
    fail_condition1 = Tracking_pool[glb_id].missing_count > missing_thred
    # fail_condition2 = (np.sqrt(np.sum(Tracking_pool[glb_id].state[0,:2,0]**2)) > 100) & (Tracking_pool[glb_id].missing_count > 0)
    if fail_condition1 :
        Off_tracking_pool[glb_id] = Tracking_pool.pop(glb_id)
        if len(Off_tracking_pool[glb_id].post_seq) == 1:
            print(Off_tracking_pool[glb_id].missing_count) 
    else:
        Tracking_pool[glb_id].state = state_cur_
        # Tracking_pool[glb_id].P = P_cur_
        Tracking_pool[glb_id].P_seq.append(Tracking_pool[glb_id].P)
        Tracking_pool[glb_id].label_seq.append(-1)
        Tracking_pool[glb_id].mea_seq.append(-1)
        Tracking_pool[glb_id].app_seq.append(Tracking_pool[glb_id].app_seq[-1])
        Tracking_pool[glb_id].post_seq.append(state_cur_)


def associate_detections(Tracking_pool,glb_id,state,app,P_post,next_label,mea_next):
    
    Tracking_pool[glb_id].state = state
    Tracking_pool[glb_id].apperance = app
    Tracking_pool[glb_id].label_seq.append(next_label)
    Tracking_pool[glb_id].mea_seq.append(mea_next)
    Tracking_pool[glb_id].post_seq.append(state)
    Tracking_pool[glb_id].app_seq.append(app)
    Tracking_pool[glb_id].P = P_post
    Tracking_pool[glb_id].P_seq.append(P_post)
    Tracking_pool[glb_id].missing_count = 0

    
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


def get_affinity_IoU(app_cur,app_next,unique_label_next,unique_label_cur,Labeling_map_cur,Labeling_map_next):
    # Union: only A or B 
    # Intersect : only A and B 
    
    Fore_next = Labeling_map_next != -1
    Fore_cur = Labeling_map_cur != -1
    Union = Fore_cur|Fore_next
    Intersect = Fore_cur & Fore_next
    Union[Intersect] = False
    
    labels_next_union,labels_cur_union = Labeling_map_next[Union],Labeling_map_cur[Union]
    pairs_union,counts_union = np.unique(np.array([labels_cur_union,labels_next_union]).T,return_counts=True,axis = 0)
    
    labels_next_intersect,labels_cur_intersect = Labeling_map_next[Intersect],Labeling_map_cur[Intersect]
    pairs_intersect,counts_intersect = np.unique(np.array([labels_cur_intersect,labels_next_intersect]).T,return_counts=True,axis = 0)

    IoU_matrix = np.zeros((unique_label_cur.shape[0],unique_label_next.shape[0]))
    dis_matrix = np.ones((unique_label_cur.shape[0],unique_label_next.shape[0]))

    for i,pair in enumerate(pairs_intersect):
        cur_label,next_label = pair[0],pair[1]
        Intersection_p = counts_intersect[i]
        A_p = counts_union[(pairs_union[:,0] == cur_label)]
        if A_p.size == 0:
            A_p = 0
        B_p = counts_union[(pairs_union[:,1] == next_label)]
        if B_p.size == 0:
            B_p = 0
        Union_p = Intersection_p + A_p + B_p
        cur_ind = unique_label_cur == cur_label
        next_ind = unique_label_next == next_label
        IoU_matrix[cur_ind,next_ind] = Intersection_p/Union_p
        dis = np.abs(app_next[next_ind,-1,0] - app_cur[cur_ind,-1,0])
        if dis > 2:
            IoU_matrix[cur_ind,next_ind] = 0
            continue
        dis_matrix[cur_ind,next_ind] = dis/2

    return 0.7*IoU_matrix + 0.3*(1 - dis_matrix) 

def get_affinity_kalman(failed_tracked_ind,new_detection_ind,state_cur_,mea_next,P_cur_):
    State_affinity =  1.5*np.ones((len(failed_tracked_ind),len(new_detection_ind)))
    for i,glb_ind in enumerate(failed_tracked_ind):
        state_pred = state_cur_[glb_ind].copy().reshape(2,-1)[:,:2]
        for j,label_ind in enumerate(new_detection_ind):
            mea = mea_next[label_ind].copy().reshape(2,-1)
            for k in range(state_pred.shape[0]):
                mal_dis = distance.mahalanobis(mea[k],state_pred[k],np.linalg.inv(P_cur_[i][k][:2,:2]))
                if mal_dis < State_affinity[i,j]:
                    State_affinity[i,j] = mal_dis
    return State_affinity

    
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

app_col_names = ['Point_Cnt','Dir_X_Bbox','Dir_Y_Bbox','Height','Length','Width','Area','Dis']
column_names_TR_2o = ['ObjectID','FrameIndex','Coord_X','Coord_Y','Coord_Z','Speed_X','Speed_Y','Speed','Longitude','Latitude','Elevation']

def get_summary_file_TR(post_seq,key,start_frame,app_seq,P_seq,T,missing_thred):
    temp = np.array(post_seq)
    temp = temp[:-missing_thred]
    temp = temp.reshape((temp.shape[0],temp.shape[1],temp.shape[2]))
    # n x 2 x 6
    temp_xy = temp[:,:,:2]
    # n x 2 x 2
#     dis_est = np.sqrt((temp_xy[:,:,0]**2 + temp_xy[:,:,1]**2))
    # n x 2 
    speed_xy = temp[:,:,2:4] * 10 
    # n x 2 x 2
    speed = np.sqrt((speed_xy[:,:,0]**2 + speed_xy[:,:,1]**2))
    # n x 2
    xyz_0 = np.concatenate([temp_xy[:,0],np.zeros(len(temp_xy)).reshape(-1,1)],axis = 1)
    xyz_1 = np.concatenate([temp_xy[:,1],np.zeros(len(temp_xy)).reshape(-1,1)],axis = 1)
    xyz = (xyz_0 + xyz_1)/2
    LLH_est = convert_LLH(xyz,T)
    est = np.concatenate([xyz_0,speed_xy[:,0],speed[:,0].reshape(-1,1),LLH_est],axis = 1)
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
    app_seq = np.array(app_seq)[:-missing_thred]
    app_seq = app_seq.reshape(-1,len(app_col_names))

    app_df = pd.DataFrame(app_seq,columns = app_col_names)

    max_length = np.percentile(np.array(app_df.Length), 80)
    app_df['Max_Length'] = max_length

    return summary,app_df


def save_result(Off_tracking_pool,ref_LLH,ref_xyz,f_path,missing_thred):

    if len(Off_tracking_pool) == 0:
        print('No Trajs Here')
    else:
        # print('Generating Traj Files...')
        T = generate_T(ref_LLH,ref_xyz)
        sums = []
        app_dfs = []
        keys = []
        start_frame = []
        lengths = []
        for key in Off_tracking_pool: 
            if len(Off_tracking_pool[key].post_seq) < (10 + missing_thred):
                continue
            sum_file,app_df = get_summary_file_TR(Off_tracking_pool[key].post_seq,
                                        key,Off_tracking_pool[key].start_frame,Off_tracking_pool[key].app_seq,Off_tracking_pool[key].P_seq,T,missing_thred) 
            sums.append(sum_file)
            app_dfs.append(app_df)
            keys.append(key)
            start_frame.append(Off_tracking_pool[key].start_frame)   
            lengths.append(len(sum_file))   
        if len(sums) != 0:
            sums = pd.concat(sums)
            app_dfs = pd.concat(app_dfs)
            sums = sums.reset_index(drop=True).astype('float64')
            app_dfs = app_dfs.reset_index(drop=True).astype('float64')

            classifier = pickle.load(open('./Classifier/Classifier.sav', 'rb'))
            X_test = np.array(app_dfs.loc[:,['Point_Cnt','Height','Max_Length','Area']])
            pred = classifier.predict(X_test)
            sums = pd.concat([sums,app_dfs,pd.DataFrame(pred.reshape(-1,1),columns=['Class'])],axis = 1)
            sums.to_csv(f_path,index = False)


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

def get_thred(temp,N = 10,d_thred = 0.1,d = 0.17,bck_n = 3):
    temp = temp.copy()
    total_sample = len(temp)
    bck_ds = []
    bck_portions = []
    repeat = 0
    while repeat < N:
        if len(temp) == 0:
            break
        sample = np.random.choice(temp,replace=False)
        ind = np.abs(temp - sample) < d
        portion = ind.sum()/total_sample
        if portion > d_thred:
            bck_portions.append(portion)
            bck_ds.append(sample)
            temp = temp[~ind]
        repeat += 1
    bck_ds = np.array(bck_ds)
    bck_portions = np.array(bck_portions)
    arg_ind = np.argsort(bck_portions)[::-1]
    bck_ds_ = bck_ds[arg_ind[:bck_n]]
    if len(bck_ds_) < bck_n:
        bck_ds_ = np.concatenate([bck_ds_,-d * np.ones(bck_n - len(bck_ds_))])
    return bck_ds_

def gen_bckmap(aggregated_maps, N, d_thred, d , bck_n):
    thred_map = np.zeros((3,32,1800))
    for i in range(thred_map.shape[1]):
        for j in range(thred_map.shape[2]):
            thred_map[:,i,j] = get_thred(aggregated_maps[:,i,j],N = N,d_thred = d_thred,d = d,bck_n = bck_n)
    return thred_map
