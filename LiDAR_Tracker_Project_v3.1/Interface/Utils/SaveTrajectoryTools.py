import numpy as np
import pandas as pd
import pickle

a = 6378137
b = 6356752.31414
e1=(a**2-b**2)/(a**2) #First eccentricity of the ellipsoid
e2=(a**2-b**2)/(b**2) #second eccentricity of the ellipsoid

app_col_names = ['Point_Cnt','Dir_X_Bbox','Dir_Y_Bbox','Height','Length','Width','Area','Dis']
column_names_TR_2o = ['ObjectID','FrameIndex','PredInd','Coord_X','Coord_Y','Coord_Z','Speed_X','Speed_Y','Speed','Longitude','Latitude','Elevation']

def save_result(Off_tracking_pool,ref_LLH,ref_xyz,f_path,start_ts,time_zone2utc):

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
            
            sum_file,app_df = get_summary_file_TR(Off_tracking_pool[key].post_seq,
                                        key,Off_tracking_pool[key].start_frame,Off_tracking_pool[key].app_seq,Off_tracking_pool[key].pred_state,T,start_ts,time_zone2utc) 
            sums.append(sum_file)
            app_dfs.append(app_df)
            keys.append(key)
            start_frame.append(Off_tracking_pool[key].start_frame)   
            lengths.append(len(sum_file))   
        
        if len(sums) != 0:
            sums = pd.concat(sums)
            app_dfs = pd.concat(app_dfs)
            sums = sums.reset_index(drop=True)
            app_dfs = app_dfs.reset_index(drop=True).astype('float64')

            classifier = pickle.load(open('./Utils/Classifier/Classifier.sav', 'rb'))
            X_test = np.array(app_dfs.loc[:,['Point_Cnt','Height','Max_Length','Area']])
            pred = classifier.predict(X_test)
            sums = pd.concat([sums,app_dfs,pd.DataFrame(pred.reshape(-1,1),columns=['Class'])],axis = 1)
            sums.to_csv(f_path,index = False)

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

def get_summary_file_TR(post_seq,key,start_frame,app_seq,pred_state,T,start_ts,time_zone2utc):
    
    temp = np.array(post_seq)
    # temp = temp[:-missing_thred]
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
    frame_ind = []
    timestamps = []
    for i in range(len(temp)):
        f = i + start_frame
        frame_ind.append('%06.0f'%f)
        timestamps.append(start_ts + (start_frame + i)*0.1)
    frame_ind = np.array(frame_ind).reshape(-1,1)
    timestamps = pd.to_datetime(timestamps,unit='s', utc=True)
    timestamps = timestamps + pd.Timedelta(time_zone2utc, unit = 'hour')
    objid = (np.ones(len(temp)) * key).astype(int).reshape(-1,1)
    pred_state = np.array(pred_state).reshape(-1,1)
    summary = np.concatenate([objid,frame_ind,pred_state,est],axis = 1)
    # obj_id,ts,x,y,z,d,s_x,s_y,s,L,L,H
    summary = pd.DataFrame(summary,columns = column_names_TR_2o)
    summary.insert(0,'Timestamp',timestamps)
    app_seq = np.array(app_seq)
    app_seq = app_seq.reshape(-1,len(app_col_names))
    app_df = pd.DataFrame(app_seq,columns = app_col_names)

    max_length = np.percentile(np.array(app_df.Length), 80)
    app_df['Max_Length'] = max_length

    return summary,app_df

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