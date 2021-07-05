import numpy as np
import open3d as op3
from scipy.optimize.optimize import main
from scipy.spatial import distance
import pandas as pd
from scipy.optimize import linear_sum_assignment

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
#sum file
col_names_ = ['X_Coord_est','Y_Coord_est','X_Len_est','Y_Len_est','Z_Len_est','X_Vel_est','Y_Vel_est','X_Acc_est','Y_Acc_est']
col_names = ['X_Coord_mea','Y_Coord_mea','X_Len_mea','Y_Len_mea','Z_Len_mea']

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

def get_params_from_detection_points(Td_map,Labeling_map,Label,Thred_map):
    
    XYZ,Labels = convert_point_cloud(Td_map,Labeling_map,Thred_map)
    pcd = op3.geometry.PointCloud()
    pcd.points = op3.utility.Vector3dVector(XYZ[Labels == Label])
    bbox = pcd.get_axis_aligned_bounding_box()
    # x,y,length,width,height 
    xylwh = np.concatenate([bbox.get_center()[:2],bbox.get_max_bound() - bbox.get_min_bound()])
    return xylwh

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
        
def associate_detections(Tracking_pool,glb_id,state,P,next_label,mea_next):
    
    Tracking_pool[glb_id].state = state
    Tracking_pool[glb_id].P = P
    Tracking_pool[glb_id].label_seq.append(next_label)
    Tracking_pool[glb_id].mea_seq.append(mea_next)
    Tracking_pool[glb_id].post_seq.append(state)
    Tracking_pool[glb_id].missing_count = 0

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

def get_affinity_mat(state,state_,P_,mea,R):
    State_affinity = np.zeros((state_.shape[0],mea.shape[0]))
    for i,s_ in enumerate(state_):
        v_ = s_.copy().flatten()
        v = state[i].copy().flatten()
        VI_spatial = P_[i][:2,:2].copy()
        VI_embed = P_[i][2:5,2:5].copy()
        v_sptial = v[:2]
        v_embed = v[2:5]
        
        for j,m in enumerate(mea):
            u = m.copy().flatten()
            d = np.sqrt(np.sum((v[:2] - u[:2])**2))
#             State_affinity[i][j] = multivariate_normal.pdf(u.flatten(),v_.flatten(),VI)
#             State_affinity[i][j] = distance.mahalanobis(u.flatten(),v_.flatten(),np.linalg.inv(VI))
            if d < 7 :
                simi_spatial = distance.mahalanobis(u[:2],v_sptial,VI_spatial)
                simi_embed = distance.mahalanobis(u[2:],v_embed,VI_embed)
                State_affinity[i][j] = 0.8*simi_spatial + 0.2*simi_embed
            else:
                State_affinity[i][j] = 1e3
            
    return State_affinity

def get_summary_file(post_seq,mea_seq):
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

if __name__ == "__main__":
    print('%06.0f.pcd'%17950)