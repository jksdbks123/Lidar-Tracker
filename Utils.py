from os import stat
import numpy as np
from numpy.core.fromnumeric import shape
import open3d as op3
from scipy.optimize.optimize import main
from scipy.spatial import distance
import pandas as pd
from scipy.optimize import linear_sum_assignment

# Kalman Filter Params

A = np.array( # x,y,l,w,h,,x',y',l',w',h',x'',y''
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
        
def associate_detections(Tracking_pool,glb_id,state,P,next_label,mea_next):
    
    Tracking_pool[glb_id].state = state
    Tracking_pool[glb_id].P = P
    Tracking_pool[glb_id].label_seq.append(next_label)
    Tracking_pool[glb_id].mea_seq.append(mea_next)
    Tracking_pool[glb_id].post_seq.append(state)
    Tracking_pool[glb_id].missing_count = 0

def state_predict_NN(state,tracking_nn):
    return tracking_nn(state).detach().numpy() # transfer to array

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

def get_affinity_mat_NN(state):
    # with _ predict, without _ is current state 
    pass

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
            if d < 7 :
                simi_spatial = distance.mahalanobis(u[:2],v_sptial,VI_spatial)
                simi_embed = distance.mahalanobis(u[2:],v_embed,VI_embed)
                State_affinity[i][j] = 0.8*simi_spatial + 0.2*simi_embed
            else:
                State_affinity[i][j] = 1e3
            
    return State_affinity

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


