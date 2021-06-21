from BfTableGenerator import *
from DDBSCAN import Raster_DBSCAN
from Utils import *
from scipy.optimize import linear_sum_assignment
import pandas as pd

class MOT():
    def __init__(self,pcap_path,background_update_frame,ending_frame,d ,thred_s ,N ,delta_thred ,step, win_size, eps, min_samples, save_pcd = True):
        """
        pcap_path : pcap file path 
        background_update_frame : background update freq
        """
        self.save_pcd = save_pcd
        self.pcap_path = pcap_path
        self.ending_frame = ending_frame
        self.traj_path = None
        self.pcd_path = None
        
        # background filtering params 
        self.d = d
        self.thred_s = thred_s
        self.N = N
        self.delta_thred = delta_thred
        self.step = step
        self.win_size = win_size
        self.eps = eps 
        self.min_samples = min_samples
        
        ###
        self.background_update_frame = background_update_frame
        self.data_collector = RansacCollector(pcap_path,background_update_frame)
        self.thred_map = None #two dimentional maps
        self.frame_gen = None
        self.db = None
        
        ###
        self.Tracking_pool = {}
        self.Off_tracking_pool = {}
        self.Global_id = 0
        
        
    def initialization(self):
        
        self.pcd_path = os.path.join(self.data_collector.output_path,'Output Pcd')
        self.traj_path = os.path.join(self.data_collector.output_path,'Output Trajs')
        if 'Output Pcd' not in os.listdir(self.data_collector.output_path ):
            os.mkdir(self.pcd_path)
        if 'Output Trajs' not in os.listdir(self.data_collector.output_path ):
            os.mkdir(self.traj_path)
            
        lidar_reader = TDmapLoader(self.pcap_path)
        frame_gen = lidar_reader.frame_gen()
        
        aggregated_maps = []
        print('Initialization...')
        for i in tqdm(range(self.background_update_frame)):
            one_frame = next(frame_gen)
            aggregated_maps.append(one_frame)
            
        aggregated_maps = np.array(aggregated_maps) # 1800 * 32 * len(ts)
        self.data_collector.aggregated_map = aggregated_maps
        self.data_collector.gen_thredmap(self.d,self.thred_s,self.N,self.delta_thred,self.step)
        self.thred_map = self.data_collector.thred_map
        
        self.db = Raster_DBSCAN(window_size=self.win_size,eps = self.eps, 
                                min_samples= self.min_samples,Td_map_szie=self.thred_map.shape)
        
        
        
    def mot_tracking(self,missing_thred,A,P_em,H,Q,R):
        
        
        
        lidar_reader = TDmapLoader(self.pcap_path)
        frame_gen = lidar_reader.frame_gen()
        self.frame_gen = frame_gen
        aggregated_maps = []
        Frame_ind = 0
        
        Td_map = next(frame_gen)
        aggregated_maps.append(Td_map)
        Foreground_map = (Td_map < self.thred_map)&(Td_map != 0)
        Labeling_map = self.db.fit_predict(Td_map= Td_map,Foreground_map=Foreground_map)
        xylwh_init,unique_label_init = extract_xylwh_by_frame(Labeling_map,Td_map,self.thred_map)

        mea_init = extract_mea_state_vec(xylwh_init)
        # m: n x 5 x 1
        state_init = np.concatenate([mea_init,np.zeros((mea_init.shape[0],A.shape[0] - H.shape[0])).reshape(mea_init.shape[0],A.shape[0] - H.shape[0],-1)],axis = 1)
        
        P_init = np.full((xylwh_init.shape[0],A.shape[0],A.shape[0]),P_em)

        for i,label in enumerate(unique_label_init):
            create_new_detection(self.Tracking_pool,self.Global_id,P_init[i],state_init[i],label,mea_init[i],Frame_ind)
            self.Global_id += 1
            
        if self.save_pcd:
            self.save_cur_pcd(Td_map,Labeling_map,self.Tracking_pool,Frame_ind)
            
        state_cur,P_cur,glb_id_cur = state_init,P_init,unique_label_init 
        pbar = tqdm(range(1,self.ending_frame))
        for Frame_ind in pbar:
            
            pbar.set_description('Tracking {} frame'.format(Frame_ind))
            if (Frame_ind % self.background_update_frame == 0) & (Frame_ind != 0): 
                aggregated_maps = np.array(aggregated_maps)
                self.data_collector.aggregated_map = aggregated_maps
                self.data_collector.gen_thredmap(self.d,self.thred_s,self.N,self.delta_thred,self.step)
                self.thred_map = self.data_collector.thred_map
                aggregated_maps = []
            """
            Extract Matrix P and State of each tracklet in Current Tracking Pool
            """
            glb_ids,P_cur,state_cur = [],[],[]
            for glb_id in self.Tracking_pool.keys():
                glb_ids.append(glb_id)
                P_cur.append(self.Tracking_pool[glb_id].P)
                state_cur.append(self.Tracking_pool[glb_id].state)
            glb_ids,P_cur,state_cur = np.array(glb_ids),np.array(P_cur),np.array(state_cur)

            state_cur_,P_cur_ = state_predict(A,Q,state_cur,P_cur) # predict next state 
            # read next data 
            Td_map = next(frame_gen)
            aggregated_maps.append(Td_map)
            Foreground_map = (Td_map < self.thred_map)&(Td_map != 0)
            Labeling_map = self.db.fit_predict(Td_map= Td_map,Foreground_map=Foreground_map)
            xylwh_next,unique_lebel_next = extract_xylwh_by_frame(Labeling_map,Td_map,self.thred_map) # read observation at next frame 
            mea_next = extract_mea_state_vec(xylwh_next)
            State_affinity = get_affinity_mat(state_cur,state_cur_,P_cur_,mea_next,R)
            associated_ind_glb,associated_ind_label = linear_sum_assignment(State_affinity)
            associated_ind_glb_,associated_ind_label_ = [],[]
            for i,ass_id in enumerate(associated_ind_glb):
                if State_affinity[ass_id,associated_ind_label[i]] < 1e3:
                    associated_ind_glb_.append(ass_id)
                    associated_ind_label_.append(associated_ind_label[i])
            associated_ind_glb,associated_ind_label = np.array(associated_ind_glb_),np.array(associated_ind_label_)
            
            """
            Failed tracking and new detections
            """
            # in a but not in b
            failed_tracked_ind = np.setdiff1d(np.arange(len(glb_ids)),associated_ind_glb) 
            
            if len(failed_tracked_ind) > 0:
                for fid in failed_tracked_ind:
                    process_fails(self.Tracking_pool,self.Off_tracking_pool,glb_ids[fid],state_cur_[fid],P_cur_[fid],missing_thred)

            new_detection_ind = np.setdiff1d(np.arange(len(unique_lebel_next)),associated_ind_label)
            if len(new_detection_ind) > 0:
                for n_id in new_detection_ind:
                    state_init = np.concatenate([mea_next[n_id], np.zeros((A.shape[0] - H.shape[0],1))])
                    create_new_detection(self.Tracking_pool,self.Global_id,P_em,state_init,unique_lebel_next[n_id],mea_next[n_id],Frame_ind)
                    self.Global_id += 1
            
                
            if len(associated_ind_glb) != 0:
                state,P = state_update(A,H,state_cur_[associated_ind_glb],P_cur_[associated_ind_glb],R,mea_next[associated_ind_label])
                glb_ids = glb_ids[associated_ind_glb]
                mea_next = mea_next[associated_ind_label]
                unique_lebel_next = unique_lebel_next[associated_ind_label]
                
                """
                Associate detections 
                """
                for i,glb_id in enumerate(glb_ids):

                    associate_detections(self.Tracking_pool,glb_id,state[i],P[i],
                                        unique_lebel_next[i],
                                        mea_next[i])
            if self.save_pcd:
                self.save_cur_pcd(Td_map,Labeling_map,self.Tracking_pool,Frame_ind)
                
        """
        Release all tracking obj into off tracking pool
        """
        release_ids = [glb_id for glb_id in self.Tracking_pool.keys()]
        for r_id in release_ids:
            self.Off_tracking_pool[r_id] = self.Tracking_pool.pop(r_id)
                    
 
    def save_result(self):
        if 'Output Trajs' not in os.listdir(self.data_collector.output_path ):
            self.traj_path = os.path.join(self.data_collector.output_path,'Output Trajs')
            os.mkdir(self.traj_path)
        
    def save_cur_pcd(self,Td_map,Labeling_map,Tracking_pool,f):
        
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
            Valid_ind =  (td_freq_map[i] != 0)&(td_freq_map[i]<self.data_collector.thred_map[i]) # None zero index
            Xs.append(X[Valid_ind])
            Ys.append(Y[Valid_ind])
            Zs.append(Z[Valid_ind])
            Labels.append(Labeling_map[i][Valid_ind])
        Xs = np.concatenate(Xs)
        Ys = np.concatenate(Ys)
        Zs = np.concatenate(Zs)
        Labels = np.concatenate(Labels).astype('int')
        Colors = np.zeros((len(Labels),3))
        for key in Tracking_pool:
            label_cur_frame = Tracking_pool[key].label_seq[-1]
            if label_cur_frame != -1:
                Colors[Labels == label_cur_frame] = color_map[key%len(color_map)]
                    
        pcd = op3.geometry.PointCloud()
        XYZ = np.concatenate([Xs.reshape(-1,1),Ys.reshape(-1,1),Zs.reshape(-1,1)],axis = 1)
        pcd.points = op3.utility.Vector3dVector(XYZ)
        pcd.colors = op3.utility.Vector3dVector(Colors)
        op3.io.write_point_cloud(os.path.join(self.pcd_path,"{}.pcd".format(f)), pcd)
        
                               
            
            
            
            
            
                
                
                
    
    
if __name__ == "__main__":
    params = {
        'd':1.2,
        'thred_s':0.3,
        'N':20,
        'delta_thred' : 1e-3,
        'step':0.1,
        'win_size':(5,13),
        'eps': 1.5,
        'min_samples':16
    }
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
    missing_thred = 7
    os.chdir(r'/Users/czhui960/Documents/Lidar/RawLidarData/US395/')
    mot = MOT(r'./US395.pcap',ending_frame=17950,background_update_frame = 2000,**params)
    mot.initialization()
    mot.mot_tracking(missing_thred,A,P,H,Q,R)

        