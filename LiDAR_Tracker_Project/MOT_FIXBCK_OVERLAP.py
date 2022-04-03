from fileinput import filename
from ntpath import join
from BfTableGenerator import *
from DDBSCAN import Raster_DBSCAN
from Utils import *
import time
import sys
import pickle
from datetime import datetime

class MOT():
    def __init__(self,input_file_path,output_file_path, win_size, eps, min_samples,bck_update_frame,
                 if_vis = False):
        """
        background_update_time : background update time (sec)
        """

        # self.output_path = output_file_path
        self.input_file_path = input_file_path
        self.traj_path = output_file_path
        self.if_vis = if_vis
        
        # params for clustering 
        self.win_size = win_size
        self.eps = eps 
        self.min_samples = min_samples
        self.bck_update_frame = bck_update_frame
        self.db = None
        self.frame_gen = TDmapLoader(input_file_path).frame_gen()
        ###
        self.thred_map = None         
        ###
        self.Tracking_pool = {}
        self.Off_tracking_pool = {}
        self.Global_id = 0
        self.Td_map_cur = None
        self.Labeling_map_cur = None
        ### Online holder
        if self.if_vis:
            self.vis = None

        
    def initialization(self):    
        
        aggregated_maps = []
        frame_gen = TDmapLoader(self.input_file_path).frame_gen()
        for i in tqdm(range(self.bck_update_frame)):
            Frame = next(frame_gen)
            if Frame is None:
                break 
            Td_map,Int_map = Frame
            aggregated_maps.append(Td_map)
        aggregated_maps = np.array(aggregated_maps)
        thred_map = gen_bckmap(aggregated_maps, d = 0.5, thred_s = 0.5, N = 20, delta_thred = 0.001, step = 0.1)
        self.thred_map = thred_map
        self.db = Raster_DBSCAN(window_size=self.win_size,eps = self.eps,min_samples= self.min_samples,Td_map_szie=self.thred_map.shape)
        print('Initialization Done')

    def mot_tracking(self): 

        if self.if_vis:
            self.vis = op3.visualization.Visualizer()
            self.vis.create_window()
        
        Frame_ind = 0
        self.frame_gen = TDmapLoader(self.input_file_path).frame_gen()
        # begin_time = time.time()
        
        while True: #Iterate Until a frame with one or more targets are detected 

            Frame = next(self.frame_gen)
            if Frame is None:
                break 
            Td_map,Intensity_map = Frame

            Foreground_map = (Td_map < self.thred_map)&(Td_map != 0)
            Labeling_map = self.db.fit_predict(Td_map= Td_map,Foreground_map=Foreground_map)
            #mea_init : n x 2 x 2 x 1
            mea_init,app_init,unique_label_init,Labeling_map = extract_xy(Labeling_map,Td_map)            
            # label here does't necessarily to be sequential from 0 - n  
            Frame_ind += 1
            if len(unique_label_init)>0:
                self.Td_map_cur = Td_map
                self.Labeling_map_cur = Labeling_map
                if self.if_vis:
                    source = self.cur_pcd(Td_map,Labeling_map,self.Tracking_pool)
                    self.vis.add_geometry(source)
                break
        

        # m: n x 2 x 2 x 1 (n objects , 2 repr point, x and y, 1 col )
        n_observed = mea_init.shape[0]
        n_repr = mea_init.shape[1]
        # mea_init.shape[2]
        n_offset_dim = 4 - mea_init.shape[2]
        state_init = np.concatenate([mea_init,np.zeros((n_observed,n_repr,n_offset_dim,1))],axis = 2)
        # s: n x 2 x 4 x 1: x,y,vx,vy
                
        for i,label in enumerate(unique_label_init):
            create_new_detection(self.Tracking_pool,self.Global_id,state_init[i],
            app_init[i],label,mea_init[i],Frame_ind)
            self.Global_id += 1
                        
        state_cur = state_init 
        cur_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        print(cur_time,'processing begin')
        aggregated_maps = []

        while True:

            """
            Extract Matrix P and State of each tracklet in Current Tracking Pool
            
            """
            Frame = next(self.frame_gen)
            if Frame is None:
                break 
            Td_map,Intensity_map = Frame
            aggregated_maps.append(Td_map)
            if Frame_ind%self.bck_update_frame == 0:
                aggregated_maps = np.array(aggregated_maps)
                self.thred_map = gen_bckmap(aggregated_maps, d = 0.5, thred_s = 0.4, N = 20, delta_thred = 0.001, step = 0.1)
                aggregated_maps = []

            time_b = time.time()
            glb_ids,state_cur,app_cur,mea_cur,unique_label_cur = [],[],[],[],[]
            for glb_id in self.Tracking_pool.keys():
                glb_ids.append(glb_id)
                state_cur.append(self.Tracking_pool[glb_id].state)
                app_cur.append(self.Tracking_pool[glb_id].apperance)
                mea_cur.append(self.Tracking_pool[glb_id].mea_seq[-1])
                unique_label_cur.append(self.Tracking_pool[glb_id].label_seq[-1])

            mea_cur = np.array(mea_cur)
            glb_ids = np.array(glb_ids)
            state_cur = np.array(state_cur)
            app_cur = np.array(app_cur)
            unique_label_cur = np.array(unique_label_cur)
            # state_cur: n x 2 x 4 x 1

            Foreground_map = (Td_map < self.thred_map)&(Td_map != 0)
            Labeling_map = self.db.fit_predict(Td_map= Td_map,Foreground_map=Foreground_map)
            mea_next,app_next,unique_label_next,Labeling_map = extract_xy(Labeling_map,Td_map)
            
             # m: n x 2 x 2 x 1 (n objects , 2 repr point, x and y, 1 col )
             # app: n x 1 x 7 x 1
             # first repr point refers to the one with lower azimuth id 
            if len(glb_ids) >0:
                if len(unique_label_next) > 0:
                    
                    State_affinity = get_affinity_mat_td(app_cur,app_next,unique_label_next,
                    unique_label_cur,self.Labeling_map_cur,Labeling_map)
                    # assiciated_ind for unique_label
                    associated_ind_cur,associated_ind_next = linear_assignment(State_affinity)
                    
                    """
                    Failed tracking and new detections
                    """
                    # in a but not in b
                    failed_tracked_ind = np.setdiff1d(np.arange(len(glb_ids)),associated_ind_cur) 
                    
                    if len(failed_tracked_ind) > 0:
                        for fid in failed_tracked_ind:
                            process_fails(self.Tracking_pool,self.Off_tracking_pool,glb_ids[fid])

                    new_detection_ind = np.setdiff1d(np.arange(len(unique_label_next)),associated_ind_next)
                    if len(new_detection_ind) > 0:
                        for n_id in new_detection_ind:
                            n_repr = mea_init.shape[1]
                            n_offset_dim = 4 - mea_init.shape[2]

                            state_init = np.concatenate([mea_next[n_id],np.zeros((n_repr,n_offset_dim,1))],axis = 1)
                            create_new_detection(self.Tracking_pool,self.Global_id,state_init,app_next[n_id],
                            unique_label_next[n_id],mea_next[n_id],Frame_ind)
                            self.Global_id += 1
                        
                    if len(associated_ind_cur) != 0:
                        glb_ids = glb_ids[associated_ind_cur]
                        state_cur = state_cur[associated_ind_cur]
                        mea_next = mea_next[associated_ind_next]
                        app_next = app_next[associated_ind_next]
                        unique_label_next = unique_label_next[associated_ind_next]
                        speed = mea_next[:,:,:2] - state_cur[:,:,:2]
                        state_cur[:,:,2:4] =  speed # calculate speed 
                        state_cur[:,:,:2] = mea_next[:,:,:2]

                        """
                        Associate detections 
                        """
                        for i,glb_id in enumerate(glb_ids):

                            associate_detections(self.Tracking_pool,glb_id,state_cur[i],app_next[i],
                                                unique_label_next[i],mea_next[i])
                else:
                    for i,glb_id in enumerate(glb_ids):
                        process_fails(self.Tracking_pool,self.Off_tracking_pool,
                                    glb_id)
            else:    
                if len(unique_label_next) > 0:
                    for n_id, mea in enumerate(mea_next):
 
                        n_repr = mea_init.shape[1]
                        n_offset_dim = 4 - mea_init.shape[2]
                        state_init = np.concatenate([mea_next[n_id],np.zeros((n_repr,n_offset_dim,1))],axis = 1)
                        create_new_detection(self.Tracking_pool,self.Global_id,state_init,
                                                app_next[n_id],unique_label_next[n_id],mea_next[n_id],Frame_ind)
                        self.Global_id += 1
            # if self.save_pcd != 'nosave':
            #     self.save_cur_pcd(Td_map,Labeling_map,self.Tracking_pool,Frame_ind)

            self.Labeling_map_cur = Labeling_map
            self.Td_map_cur = Td_map
            Frame_ind += 1 
            time_c = time.time()
            if self.if_vis:
                pcd = self.cur_pcd(Td_map,Labeling_map,self.Tracking_pool)
                source.points = pcd.points
                source.colors = pcd.colors
                self.vis.update_geometry(source)
                self.vis.poll_events()
                self.vis.update_renderer()   
            time_d = time.time()
            sys.stdout.write('\rProcessing Time: {}'.format(round((time_c - time_b) * 1000,2)))
            sys.stdout.flush()
        if self.if_vis:
            self.vis.destroy_window() 

        release_ids = [glb_id for glb_id in self.Tracking_pool.keys()]
        for r_id in release_ids:
            self.Off_tracking_pool[r_id] = self.Tracking_pool.pop(r_id)
        
        file_name = '{}.pickle'.format(cur_time) 
        with open(os.path.join(self.traj_path,file_name), 'wb') as handle:
            pickle.dump(self.Off_tracking_pool, handle, protocol=pickle.HIGHEST_PROTOCOL)


            
    def cur_pcd(self,Td_map,Labeling_map,Tracking_pool):
        td_freq_map = Td_map
        Xs = []
        Ys = []
        Zs = []

        Labels = []
        for i in range(td_freq_map.shape[0]):
            longitudes = theta[i] * np.pi / 180
            latitudes = azimuths * np.pi / 180 
            hypotenuses = td_freq_map[i] * np.cos(longitudes)
            X = hypotenuses * np.sin(latitudes)
            Y = hypotenuses * np.cos(latitudes)
            Z = td_freq_map[i] * np.sin(longitudes)
            Valid_ind = td_freq_map[i] != 0 
            Xs.append(X[Valid_ind])
            Ys.append(Y[Valid_ind])
            Zs.append(Z[Valid_ind])
            Labels.append(Labeling_map[i][Valid_ind])

        Xs = np.concatenate(Xs)
        Ys = np.concatenate(Ys)
        Zs = np.concatenate(Zs)
            
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

        return pcd


if __name__ == "__main__":
    params = {
        "win_size":[5,13],
        "eps" : 1.5,
        "min_samples" : 12,
        "bck_update_frame":2000,
        "if_vis":False
        }

    output_file_path = r'D:/Test'
    input_file_path = r'D:/LiDAR_Data/MidTown/California/2021-12-8-18-0-0.pcap'
    mot = MOT(input_file_path,output_file_path,**params)
    mot.initialization()
    mot.mot_tracking()
    # mot.mot_tracking(A,P,H,Q,R)
    # mot.save_result(ref_LLH,ref_xyz)
