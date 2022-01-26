from BfTableGenerator import *
from DDBSCAN import Raster_DBSCAN
from Utils import *

class MOT():
    def __init__(self,pcap_path,output_file_path,background_update_frame,ending_frame,d ,thred_s ,N ,
                 delta_thred ,step, win_size, eps, min_samples, missing_thred,
                 save_pcd = None, save_Azimuth_Laser_info = False,save_TD_Map = False,save_Labeling_map = False, result_type = 'split'):
        """
        pcap_path : pcap file path 
        background_update_frame : background update freq
        """
        self.save_pcd = save_pcd
        self.save_Azimuth_Laser_info = save_Azimuth_Laser_info
        self.save_TDMap = save_TD_Map
        self.save_LabelingMap = save_Labeling_map
        self.pcap_path = pcap_path
        self.ending_frame = ending_frame
        self.traj_path = None
        self.pcd_path = None
        self.Azimuth_Laser_info_path = None
        self.result_type = result_type
        
        
        # background filtering params 
        self.d = d
        self.thred_s = thred_s
        self.N = N
        self.delta_thred = delta_thred
        self.step = step
        self.win_size = win_size
        self.eps = eps 
        self.min_samples = min_samples
        self.missing_thred = missing_thred
        
        ###
        self.background_update_frame = None
        self.data_collector = RansacCollector(pcap_path,output_file_path,self.background_update_frame)
        self.thred_map = None #two dimentional maps
        self.frame_gen = None
        
        ###
        self.Tracking_pool = {}
        self.Off_tracking_pool = {}
        self.Global_id = 0
        
        
    def initialization(self,bck_map):
        
        self.pcd_path = os.path.join(self.data_collector.output_path,'OutputPcd')
        self.traj_path = os.path.join(self.data_collector.output_path,'OutputTrajs')
        self.Azimuth_Laser_info_path = os.path.join(self.data_collector.output_path,'OutputAzimuthLaserInfo')
        self.Labeling_map_path =  os.path.join(self.data_collector.output_path,'LabelingMaps')
        self.Td_map_path =  os.path.join(self.data_collector.output_path,'TDMaps')

        if 'OutputPcd' not in os.listdir(self.data_collector.output_path ):
            os.mkdir(self.pcd_path)
        if 'OutputTrajs' not in os.listdir(self.data_collector.output_path):
            os.mkdir(self.traj_path)
        if 'OutputAzimuthLaserInfo' not in os.listdir(self.data_collector.output_path):
            os.mkdir(self.Azimuth_Laser_info_path)
        if 'LabelingMaps' not in os.listdir(self.data_collector.output_path):
            os.mkdir(self.Labeling_map_path)
        if 'TDMaps' not in os.listdir(self.data_collector.output_path):
            os.mkdir(self.Td_map_path)   

        self.thred_map = bck_map
        self.db = Raster_DBSCAN(window_size=self.win_size,eps = self.eps, 
                                min_samples= self.min_samples,Td_map_szie=self.thred_map.shape)     
        
    def mot_tracking(self,A,P,H,Q,R):
        
        missing_thred = self.missing_thred
        lidar_reader = TDmapLoader(self.pcap_path)
        frame_gen = lidar_reader.frame_gen()
        self.frame_gen = frame_gen
        Frame_ind_init = 0
        
        while True: #Iterate Until a frame with one or more targets are detected 
            Td_map = next(frame_gen)
            Foreground_map = (Td_map < self.thred_map)&(Td_map != 0)
            Labeling_map = self.db.fit_predict(Td_map= Td_map,Foreground_map=Foreground_map)
            Background_map = (Td_map >= self.thred_map)&(Td_map != 0)
            #mea_init : n x 2 x 2 x 1
            mea_init,app_init,unique_label_init,Labeling_map = extract_xy(Labeling_map,Td_map)
            
            if self.save_pcd != 'nosave':
                self.save_cur_pcd(Td_map,Labeling_map,self.Tracking_pool,Frame_ind_init)
            if self.save_LabelingMap:
                self.save_Labeling_map(Labeling_map,Frame_ind_init)
            if self.save_TDMap:
                self.save_TD_map(Td_map,Frame_ind_init)
            Frame_ind_init += 1
            if len(unique_label_init)>0:
                break
        # m: n x 2 x 2 x 1 (n objects , 2 repr point, x and y, 1 col )
        n_observed = mea_init.shape[0]
        n_repr = mea_init.shape[1]
        # mea_init.shape[2]
        n_offset_dim = A.shape[0] - mea_init.shape[2]
        state_init = np.concatenate([mea_init,np.zeros((n_observed,n_repr,n_offset_dim,1))],axis = 2)
        # s: n x 2 x 4 x 1
                
        for i,label in enumerate(unique_label_init):
            create_new_detection_NEAREST(self.Tracking_pool,self.Global_id,state_init[i],app_init[i],label,mea_init[i],Frame_ind_init)
            self.Global_id += 1
                        
        state_cur = state_init 
        
        pbar = tqdm(range(Frame_ind_init,self.ending_frame))
        
        for Frame_ind in pbar:
            pbar.set_description('Tracking {} frame'.format(Frame_ind))
            """
            Extract Matrix P and State of each tracklet in Current Tracking Pool
            
            """
            glb_ids,state_cur,app_cur, = [],[],[]
            for glb_id in self.Tracking_pool.keys():
                glb_ids.append(glb_id)
                state_cur.append(self.Tracking_pool[glb_id].state)
                app_cur.append(self.Tracking_pool[glb_id].apperance)

            glb_ids,state_cur,app_cur = np.array(glb_ids),np.array(state_cur),np.array(app_cur)
            # P_cur: n x 2 x 4 x 4
            # state_cur: n x 2 x 4 x 1
            # heading_vecs: n x 2 x 2 x 1
            # read next data 
            Td_map = next(frame_gen)
            Foreground_map = (Td_map < self.thred_map)&(Td_map != 0)
            Labeling_map = self.db.fit_predict(Td_map= Td_map,Foreground_map=Foreground_map)
            Background_map = (Td_map >= self.thred_map)&(Td_map != 0)
            mea_next,app_next,unique_label_next,Labeling_map = extract_xy_interval_merging_TR(Labeling_map,Td_map,Background_map)
             # m: n x 2 x 2 x 1 (n objects , 2 repr point, x and y, 1 col )
             # app: n x 1 x 7 x 1
             # first repr point refers to the one with lower azimuth id 
            if len(glb_ids) >0:
                if len(unique_label_next) > 0:

                    State_affinity = get_affinity_dis_box_TR_NEAREST(state_cur,mea_next,app_next,app_cur)
                    associated_ind_glb,associated_ind_label = linear_assignment_modified_dis(State_affinity)
                    
                    """
                    Failed tracking and new detections
                    """
                    # in a but not in b
                    failed_tracked_ind = np.setdiff1d(np.arange(len(glb_ids)),associated_ind_glb) 
                    
                    if len(failed_tracked_ind) > 0:
                        for fid in failed_tracked_ind:
                            process_fails_NEAREST(self.Tracking_pool,self.Off_tracking_pool,
                                        glb_ids[fid])

                    new_detection_ind = np.setdiff1d(np.arange(len(unique_label_next)),associated_ind_label)
                    if len(new_detection_ind) > 0:
                        for n_id in new_detection_ind:
                            n_repr = mea_init.shape[1]
                            n_offset_dim = A.shape[0] - mea_init.shape[2]
                            state_init = np.concatenate([mea_next[n_id],np.zeros((n_repr,n_offset_dim,1))],axis = 1)
                            
                            create_new_detection_NEAREST(self.Tracking_pool,self.Global_id,state_init,
                                                app_next[n_id],unique_label_next[n_id],mea_next[n_id],Frame_ind)
                            self.Global_id += 1
                        
                    if len(associated_ind_glb) != 0:

                        state_cur = state_cur[associated_ind_glb]
                        
                        glb_ids = glb_ids[associated_ind_glb]
                        mea_next = mea_next[associated_ind_label]
                        app_next = app_next[associated_ind_label]
                        unique_label_next = unique_label_next[associated_ind_label]
                        speed = mea_next[:,:,:2] - state_cur[:,:,:2]
                        state_cur[:,:,2:4] =  speed # calculate speed 
                        state_cur[:,:,:2] = mea_next[:,:,:2]

                        """
                        Associate detections 
                        """
                        for i,glb_id in enumerate(glb_ids):
                            associate_detections_NEAREST(self.Tracking_pool,glb_id,state_cur[i],app_next[i],unique_label_next[i],mea_next[i])
                else:
                    for i,glb_id in enumerate(glb_ids):
                        process_fails_NEAREST(self.Tracking_pool,self.Off_tracking_pool,
                                        glb_ids[glb_id])

            else:    
                if len(unique_label_next) > 0:
                    for n_id in range(len(mea_next)):
                        
                        n_repr = mea_init.shape[1]
                        n_offset_dim = A.shape[0] - mea_init.shape[2]
                        state_init = np.concatenate([mea_next[n_id],np.zeros((n_repr,n_offset_dim,1))],axis = 1)

                        create_new_detection_NEAREST(self.Tracking_pool,self.Global_id,state_init,
                                            app_next[n_id],unique_label_next[n_id],mea_next[n_id],Frame_ind)
                        self.Global_id += 1

            if self.save_pcd != 'nosave':
                self.save_cur_pcd(Td_map,Labeling_map,self.Tracking_pool,Frame_ind)
            if self.save_LabelingMap:
                self.save_Labeling_map(Labeling_map,Frame_ind)
            if self.save_TDMap:
                self.save_TD_map(Td_map,Frame_ind)
                
        """
        Release all tracking obj into off tracking pool
        """
        release_ids = [glb_id for glb_id in self.Tracking_pool.keys()]
        for r_id in release_ids:
            self.Off_tracking_pool[r_id] = self.Tracking_pool.pop(r_id)
                    
    def save_TD_map(self,Td_map,f):
        np.save(os.path.join(self.Td_map_path,'%06.0f.npy'%f),Td_map)
    def save_Labeling_map(self,Labeling_map,f):
        np.save(os.path.join(self.Labeling_map_path,'%06.0f.npy'%f),Labeling_map)

    def save_result(self,ref_LLH,ref_xyz):
        
        if 'OutputTrajs' not in os.listdir(self.data_collector.output_path ):
            self.traj_path = os.path.join(self.data_collector.output_path,'OutputTrajs')
            os.mkdir(self.traj_path)
        
        print('Generating Traj Files...')
        T = generate_T(ref_LLH,ref_xyz)

        if self.result_type == 'split':
            keys = []
            start_frame = []
            lengths = []
            for key in tqdm(self.Off_tracking_pool):  
                sum_file = get_summary_file_split(self.Off_tracking_pool[key].post_seq,self.Off_tracking_pool[key].mea_seq)
                sum_file.to_csv(self.traj_path + '/{}.csv'.format(key),index = False)
                keys.append(key)
                start_frame.append(self.Off_tracking_pool[key].start_frame)
                lengths.append(len(self.Off_tracking_pool[key].post_seq))
                
            pd.DataFrame({
                'Glb_ID':keys,
                'Start_Frame':start_frame,
                'Len':lengths
            }).to_csv(self.data_collector.output_path+'/Summary.csv')
        else:
            sums_0 = []
            sums_1 = []
            app_dfs = []
            keys = []
            start_frame = []
            lengths = []
            for key in tqdm(self.Off_tracking_pool):  

                sum_file_0,sum_file_1,app_df = get_summary_file_TR(self.Off_tracking_pool[key].post_seq,self.Off_tracking_pool[key].mea_seq,
                                            key,self.Off_tracking_pool[key].start_frame,self.Off_tracking_pool[key].app_seq,self.missing_thred,T) 
                sums_0.append(sum_file_0)
                sums_1.append(sum_file_1)
                app_dfs.append(app_df)
                keys.append(key)
                start_frame.append(self.Off_tracking_pool[key].start_frame)   
                lengths.append(len(sum_file_0))   

            sums_0 = pd.concat(sums_0)
            sums_1 = pd.concat(sums_1)
            app_dfs = pd.concat(app_dfs)
            sums_0 = sums_0.reset_index(drop=True).astype('float32')
            sums_1 = sums_1.reset_index(drop=True).astype('float32')
            app_dfs = app_dfs.reset_index(drop=True).astype('float32')


            classifier = pickle.load(open('./Classifier/Classifier.sav', 'rb'))
            X_test = np.array(app_dfs.loc[:,['Point_Cnt','Height','Length','Area']])
            pred = classifier.predict(X_test)
            sums_0 = pd.concat([sums_0,app_dfs,pd.DataFrame(pred.reshape(-1,1),columns=['Class'])],axis = 1)
            sums_1 = pd.concat([sums_1,app_dfs,pd.DataFrame(pred.reshape(-1,1),columns=['Class'])],axis = 1)
            sums_0.to_csv(self.traj_path + '/Trajctories_0.csv',index = False)
            sums_1.to_csv(self.traj_path + '/Trajctories_1.csv',index = False)
            pd.DataFrame({
                'Glb_ID':keys,
                'Start_Frame':start_frame,
                'Len':lengths
            }).to_csv(self.data_collector.output_path+'/Summary.csv')

            
    def save_cur_pcd(self,Td_map,Labeling_map,Tracking_pool,f):
        
        td_freq_map = Td_map
        Xs = []
        Ys = []
        Zs = []
        if self.save_Azimuth_Laser_info:
            Azimuth_channels = []
            Laser_ids = []
            Distances = []

        Labels = []
        for i in range(td_freq_map.shape[0]):
            longitudes = theta[i] * np.pi / 180
            latitudes = azimuths * np.pi / 180 
            hypotenuses = td_freq_map[i] * np.cos(longitudes)
            X = hypotenuses * np.sin(latitudes)
            Y = hypotenuses * np.cos(latitudes)
            Z = td_freq_map[i] * np.sin(longitudes)
            if self.save_pcd == 'Filtered':
                Valid_ind = (td_freq_map[i] != 0)&(td_freq_map[i]<self.thred_map[i]) # None zero index
            else:
                Valid_ind = td_freq_map[i] != 0 
            Xs.append(X[Valid_ind])
            Ys.append(Y[Valid_ind])
            Zs.append(Z[Valid_ind])
            Labels.append(Labeling_map[i][Valid_ind])
            if self.save_Azimuth_Laser_info:
                Azimuth_channels.append(np.where(Valid_ind)[0])
                Laser_ids.append(i*np.ones(Valid_ind.sum()).astype('int'))
                Distances.append(td_freq_map[i][Valid_ind])
                
        Xs = np.concatenate(Xs)
        Ys = np.concatenate(Ys)
        Zs = np.concatenate(Zs)
        if self.save_Azimuth_Laser_info :
            Azimuth_channels = np.concatenate(Azimuth_channels)
            Laser_ids = np.concatenate(Laser_ids)
            Distances = np.concatenate(Distances)
            LA_info = np.concatenate([Laser_ids.reshape(-1,1),Azimuth_channels.reshape(-1,1),Distances.reshape(-1,1)],axis = 1)
            
        Labels = np.concatenate(Labels).astype('int')
        Colors = np.full((len(Labels),3),np.array([[153,153,153]])/256)
        for key in Tracking_pool:
            label_cur_frame = Tracking_pool[key].label_seq[-1]
            if label_cur_frame != -1:
                Colors[Labels == label_cur_frame] = color_map[key%len(color_map)]
                    
        pcd = op3.geometry.PointCloud()
        XYZ = np.concatenate([Xs.reshape(-1,1),Ys.reshape(-1,1),Zs.reshape(-1,1)],axis = 1)
        pcd.points = op3.utility.Vector3dVector(XYZ)
        pcd.colors = op3.utility.Vector3dVector(Colors)
        op3.io.write_point_cloud(os.path.join(self.pcd_path,'%06.0f.pcd'%f), pcd)



if __name__ == "__main__":
    
    input_path = 'D:\LiDAR_Data\MidTown\California'
    dir_lis = os.listdir(input_path)
    pcap_path = 'None'
    for f in dir_lis:
        if 'pcap' in f.split('.'):
            pcap_path = os.path.join(input_path,f)
    if pcap_path == 'None':
        print('Pcap file is not detected')
    output_file_path = 'D:\LiDAR_Data\MidTown\California'
    config_path = os.path.join(input_path,'config.json')
    ref_LLH_path,ref_xyz_path = os.path.join(input_path,'LLE_ref.csv'),os.path.join(input_path,'xyz_ref.csv')
    ref_LLH,ref_xyz = np.array(pd.read_csv(ref_LLH_path)),np.array(pd.read_csv(ref_xyz_path))
    ref_LLH[:,[0,1]] = ref_LLH[:,[0,1]] * np.pi/180
    ref_LLH[:,2] = ref_LLH[:,2]/3.2808

    with open(config_path) as f:
        params = json.load(f)

    print(params)
    mot = MOT(pcap_path,output_file_path,**params)
    bck_map_path = os.path.join(input_path,'bck_map.npy')
    bck_map = np.load(bck_map_path)
    mot.initialization(bck_map)
    mot.mot_tracking(A,P,H,Q,R)
    mot.save_result(ref_LLH,ref_xyz)
