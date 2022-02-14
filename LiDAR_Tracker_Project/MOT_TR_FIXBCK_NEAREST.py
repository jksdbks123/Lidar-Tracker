from BfTableGenerator import *
from DDBSCAN import Raster_DBSCAN
from Utils import *

class MOT():
    def __init__(self,pcap_path,output_file_path,ending_frame,win_size, eps, min_samples):
        """
        pcap_path : pcap file path 
        background_update_frame : background update freq
        """
        self.pcap_path = pcap_path
        self.ending_frame = ending_frame
        self.traj_path = output_file_path
        self.win_size = win_size
        self.eps = eps 
        self.min_samples = min_samples        
        ###
        self.background_update_frame = None
        self.thred_map = None #two dimentional maps
        self.frame_gen = None

        ###
        self.Tracking_pool = {}
        self.Off_tracking_pool = {}
        self.Global_id = 0
        
        
    def initialization(self,bck_map):
        

        self.thred_map = bck_map
        self.db = Raster_DBSCAN(window_size=self.win_size,eps = self.eps, 
                                min_samples= self.min_samples,Td_map_szie=self.thred_map.shape)     

    def mot_tracking(self,A,Plane_model):
        
        lidar_reader = TDmapLoader(self.pcap_path)
        frame_gen = lidar_reader.frame_gen()
        self.frame_gen = frame_gen
        Frame_ind_init = 0
        Termination_sign = False

        while True: #Iterate Until a frame with one or more targets are detected 
            Td_map = next(frame_gen)
            if Td_map is None:
                Termination_sign = True
                mea_init = None
                break
            Foreground_map = (Td_map < self.thred_map)&(Td_map != 0)
            Labeling_map = self.db.fit_predict(Td_map= Td_map,Foreground_map=Foreground_map)
            #mea_init : n x 2 x 2 x 1
            mea_init,app_init,unique_label_init,Labeling_map = extract_xy(Labeling_map,Td_map,Plane_model)
            Frame_ind_init += 1
            if len(unique_label_init)>0:
                break
        if Termination_sign is not True:
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
            
            # pbar = tqdm(range(Frame_ind_init,self.ending_frame))
                            # pbar.set_description('Tracking {} frame'.format(Frame_ind))

            for Frame_ind in range(Frame_ind_init,self.ending_frame):
                # pbar.set_description('Tracking {} frame'.format(Frame_ind))
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
                if Td_map is None:
                    Termination_sign = True
                    break
                Foreground_map = (Td_map < self.thred_map)&(Td_map != 0)
                Labeling_map = self.db.fit_predict(Td_map= Td_map,Foreground_map=Foreground_map)
                mea_next,app_next,unique_label_next,Labeling_map = extract_xy(Labeling_map,Td_map,Plane_model)
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
                                            glb_id)

                else:    
                    if len(unique_label_next) > 0:
                        for n_id in range(len(mea_next)):
                            
                            n_repr = mea_init.shape[1]
                            n_offset_dim = A.shape[0] - mea_init.shape[2]
                            state_init = np.concatenate([mea_next[n_id],np.zeros((n_repr,n_offset_dim,1))],axis = 1)

                            create_new_detection_NEAREST(self.Tracking_pool,self.Global_id,state_init,
                                                app_next[n_id],unique_label_next[n_id],mea_next[n_id],Frame_ind)
                            self.Global_id += 1

        """
        Release all tracking obj into off tracking pool
        """
        self.exit_tracking()

    def exit_tracking(self):
        release_ids = [glb_id for glb_id in self.Tracking_pool.keys()]
        for r_id in release_ids:
            self.Off_tracking_pool[r_id] = self.Tracking_pool.pop(r_id)       

    def save_result(self,ref_LLH,ref_xyz):
        

        if len(self.Off_tracking_pool) == 0:
            print('No Trajs Here')
        else:
            # print('Generating Traj Files...')
            T = generate_T(ref_LLH,ref_xyz)
            sums = []
            app_dfs = []
            keys = []
            start_frame = []
            lengths = []
            for key in self.Off_tracking_pool:  
                sum_file,app_df = get_summary_file_TR(self.Off_tracking_pool[key].post_seq,
                                            key,self.Off_tracking_pool[key].start_frame,self.Off_tracking_pool[key].app_seq,T) 
                sums.append(sum_file)
                app_dfs.append(app_df)
                keys.append(key)
                start_frame.append(self.Off_tracking_pool[key].start_frame)   
                lengths.append(len(sum_file))   

            sums = pd.concat(sums)
            app_dfs = pd.concat(app_dfs)
            sums = sums.reset_index(drop=True).astype('float64')
            app_dfs = app_dfs.reset_index(drop=True).astype('float64')

            classifier = pickle.load(open('./Classifier/Classifier.sav', 'rb'))
            X_test = np.array(app_dfs.loc[:,['Point_Cnt','Height','Length','Area']])
            pred = classifier.predict(X_test)
            sums = pd.concat([sums,app_dfs,pd.DataFrame(pred.reshape(-1,1),columns=['Class'])],axis = 1)
            f_name = self.pcap_path.split('\\')[-1].split('.')[-2] + '.csv'
            sums.to_csv(os.path.join(self.traj_path,f_name),index = False)
    
            



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
