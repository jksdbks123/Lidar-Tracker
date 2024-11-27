from tkinter import SEL
from BfTableGenerator import *
from DDBSCAN import Raster_DBSCAN
from Utils import *
import time
import sys
from datetime import datetime

class MOT():
    def __init__(self,input_file_path,output_file_path, win_size, eps, min_samples,bck_update_frame,N = 10,d_thred = 0.08,bck_n = 3,missing_thred = 10,bck_radius = 0.9
                 ,if_save_pcd = False ):
        """
        background_update_time : background update time (sec)
        """

        # self.output_path = output_file_path
        self.input_file_path = input_file_path
        self.traj_path = output_file_path
        # if no data in pcap, then it's False
        self.if_pcap_valid = True 
        # params for clustering and background sampling
        self.win_size = win_size
        self.eps = eps 
        self.min_samples = min_samples
        self.bck_update_frame = bck_update_frame
        self.d_thred = d_thred 
        self.N = N
        self.bck_n = bck_n
        self.bck_radius = bck_radius
        self.db = Raster_DBSCAN(window_size=self.win_size,eps = self.eps,min_samples= self.min_samples,Td_map_szie=(32,1800))
        ###
        self.thred_map = None     
        self.start_timestamp = 0    
        ###
        self.Tracking_pool = {}
        self.Off_tracking_pool = {}
        self.Global_id = 0
        self.Td_map_cur = None
        self.Labeling_map_cur = None
        self.missing_thred = missing_thred
        ### Online holder
        self.frame_gen = None
        self.pcd_list = []




        
    def initialization(self,aggregated_maps):    
        # record start unix timestamp 
        temp = []
        print(len(aggregated_maps))
        for i in range(self.bck_update_frame):
            temp.append(aggregated_maps[i])
        print('Generating Bck')
        thred_map = gen_bckmap(np.array(temp), N = self.N, d_thred = self.d_thred, bck_n = self.bck_n)
        print('Bck Done')
        self.thred_map = thred_map

    def test(self,aggregated_map,event,TrackingStatus):
        frame = 0 
        while True:
            if event.is_set():
                print('Terminated')
                break
            print(len(aggregated_map))
            TrackingStatus.set('Tracked Frames:{}'.format(frame))
            frame+=1
            time.sleep(1)

    def mot_tracking(self,aggregated_maps,event,TrackingStatus): 
        """
        Init Tracker
        """

        Frame_ind = 0
        
        while True: 
        #Iterate Until a frame with one or more targets are detected 

            Td_map = aggregated_maps[Frame_ind]
            if event.is_set():
                break 
            Foreground_map = ~(np.abs(Td_map - self.thred_map) <= self.bck_radius).any(axis = 0)
            Labeling_map = self.db.fit_predict(Td_map= Td_map,Foreground_map=Foreground_map)
            #mea_init : n x 2 x 2 x 1
            mea_init,app_init,unique_label_init,Labeling_map = extract_xy(Labeling_map,Td_map)            
            # label here does't necessarily to be sequential from 0 - n  
            Frame_ind += 1
            TrackingStatus.set('Tracked Frames:{}'.format(Frame_ind))
            if len(unique_label_init)>0:
                self.Td_map_cur = Td_map
                self.Labeling_map_cur = Labeling_map
                break
        
        # m: n x 2 x 2 x 1 (n objects , 2 repr point, x and y, 1 col )
        n_observed = mea_init.shape[0]
        n_repr = mea_init.shape[1]
        n_offset_dim = A.shape[0] - mea_init.shape[2]
        state_init = np.concatenate([mea_init,np.zeros((n_observed,n_repr,n_offset_dim,1))],axis = 2)
        P_init = np.full((n_observed,2,P_em.shape[0],P_em.shape[1]),P_em)

        # s: n x 2 x 4 x 1: x,y,vx,vy
                
        for i,label in enumerate(unique_label_init):
            create_new_detection(self.Tracking_pool,self.Global_id,state_init[i],
            app_init[i],label,mea_init[i],P_init[i],Frame_ind)
            self.Global_id += 1
                        
        state_cur,P_cur = state_init,P_init 

        while True:

            """
            Extract Matrix P and State of each tracklet in Current Tracking Pool
            
            """
            Td_map = aggregated_maps[Frame_ind]
            if event.is_set():
                break
            if Frame_ind%self.bck_update_frame == 0:
                temp = []
                for i in range(Frame_ind - self.bck_update_frame, Frame_ind):
                    temp.append(aggregated_maps[i])
                print('Gen Bck')
                thred_map = gen_bckmap(np.array(temp), N = self.N, d_thred = self.d_thred, bck_n = self.bck_n)
                self.thred_map = thred_map

            glb_ids,state_cur,app_cur,unique_label_cur,P_cur = [],[],[],[],[]
            for glb_id in self.Tracking_pool.keys():
                glb_ids.append(glb_id)
                P_cur.append(self.Tracking_pool[glb_id].P)
                state_cur.append(self.Tracking_pool[glb_id].state)
                app_cur.append(self.Tracking_pool[glb_id].apperance)
                # mea_cur.append(self.Tracking_pool[glb_id].mea_seq[-1])
                unique_label_cur.append(self.Tracking_pool[glb_id].label_seq[-1])

            # mea_cur = np.array(mea_cur)
            glb_ids = np.array(glb_ids)
            state_cur = np.array(state_cur)
            app_cur = np.array(app_cur)
            unique_label_cur = np.array(unique_label_cur)
            P_cur = np.array(P_cur)
            # state_cur: n x 2 x 4 x 1

            # Foreground_map = (Td_map < self.thred_map)&(Td_map != 0)
            Foreground_map = ~(np.abs(Td_map - self.thred_map) <= self.bck_radius).any(axis = 0)
            Labeling_map = self.db.fit_predict(Td_map= Td_map,Foreground_map=Foreground_map)
            
             # m: n x 2 x 2 x 1 (n objects , 2 repr point, x and y, 1 col )
             # app: n x 1 x 7 x 1
             # first repr point refers to the one with lower azimuth id 
            mea_next,app_next,unique_label_next,Labeling_map = extract_xy(Labeling_map,Td_map)

            if len(glb_ids) >0:
                # merging included
                
                state_cur_,P_cur_ = state_predict(A,Q,state_cur,P_cur)

                if len(unique_label_next) > 0:
                    # app_next : n x 7 x 1
                    State_affinity = get_affinity_IoU(app_cur,app_next,unique_label_next,
                     unique_label_cur,self.Labeling_map_cur,Labeling_map)
                    
                    # assiciated_ind for unique_label_next and unique_label_cur
                    associated_ind_cur,associated_ind_next = linear_assignment(State_affinity)
                    # in a but not in b
                    failed_tracked_ind = np.setdiff1d(np.arange(len(glb_ids)),associated_ind_cur) 
                    new_detection_ind = np.setdiff1d(np.arange(len(unique_label_next)),associated_ind_next)
                    if ( len(failed_tracked_ind) > 0 ) & (len(new_detection_ind) > 0):
                        State_affinity_kalman = get_affinity_kalman(failed_tracked_ind,new_detection_ind,state_cur_,mea_next,P_cur_)
                        failed_tracked_ind_,new_detection_ind_ = linear_assignment_kalman(State_affinity_kalman)

                        if len(failed_tracked_ind_) > 0:
                            associated_ind_cur = np.concatenate([associated_ind_cur,failed_tracked_ind[failed_tracked_ind_]]).astype(int)
                            associated_ind_next = np.concatenate([associated_ind_next,new_detection_ind[new_detection_ind_]]).astype(int)

                    """
                    Failed tracking and new detections
                    """
                    failed_tracked_ind = np.setdiff1d(np.arange(len(glb_ids)),associated_ind_cur) 
                    if len(failed_tracked_ind) > 0:
                        for fid in failed_tracked_ind:
                            process_fails(self.Tracking_pool,self.Off_tracking_pool,glb_ids[fid],state_cur_[fid],P_cur_[fid],self.missing_thred)

                    new_detection_ind = np.setdiff1d(np.arange(len(unique_label_next)),associated_ind_next)
                    if len(new_detection_ind) > 0:
                        for n_id in new_detection_ind:
                            n_repr = mea_init.shape[1]
                            n_offset_dim = 4 - mea_init.shape[2]
                            state_init = np.concatenate([mea_next[n_id],np.zeros((n_repr,n_offset_dim,1))],axis = 1)
                            P_init = np.full((2,P_em.shape[0],P_em.shape[1]),P_em)
                            create_new_detection(self.Tracking_pool,self.Global_id,state_init,app_next[n_id],
                            unique_label_next[n_id],mea_next[n_id],P_init,Frame_ind)
                            self.Global_id += 1

                    if len(associated_ind_cur) != 0:
                        # print(associated_ind_cur,associated_ind_next)
                        state_cur,P_post = state_update(A,H,state_cur_[associated_ind_cur],P_cur_[associated_ind_cur],R,mea_next[associated_ind_next])
                        glb_ids = glb_ids[associated_ind_cur]
                        # state_cur = state_post[associated_ind_cur]
                        mea_next = mea_next[associated_ind_next]
                        app_next = app_next[associated_ind_next]
                        unique_label_next = unique_label_next[associated_ind_next]
                        """
                        Associate detections 
                        """
                        for i,glb_id in enumerate(glb_ids):
                            associate_detections(self.Tracking_pool,glb_id,state_cur[i],app_next[i],P_post[i],
                                                unique_label_next[i],mea_next[i])
                else:
                    for i,glb_id in enumerate(glb_ids):

                        process_fails(self.Tracking_pool,self.Off_tracking_pool,
                                    glb_id,state_cur_[i],P_cur_[i],self.missing_thred)
            else:    
                if len(unique_label_next) > 0:
                    for n_id, mea in enumerate(mea_next):
 
                        n_repr = mea_init.shape[1]
                        n_offset_dim = 4 - mea_init.shape[2]
                        state_init = np.concatenate([mea_next[n_id],np.zeros((n_repr,n_offset_dim,1))],axis = 1)
                        P_init = np.full((2,P_em.shape[0],P_em.shape[1]),P_em)
                        create_new_detection(self.Tracking_pool,self.Global_id,state_init,
                                                app_next[n_id],unique_label_next[n_id],mea_next[n_id],P_init,Frame_ind)
                        self.Global_id += 1

                        
            # if self.save_pcd != 'nosave':
            #     self.save_cur_pcd(Td_map,Labeling_map,self.Tracking_pool,Frame_ind)
            TrackingStatus.set('Tracked Frames:{}'.format(Frame_ind))
            self.Labeling_map_cur = Labeling_map
            self.Td_map_cur = Td_map
            Frame_ind += 1 

        release_ids = [glb_id for glb_id in self.Tracking_pool.keys()]
        for r_id in release_ids:
            self.Off_tracking_pool[r_id] = self.Tracking_pool.pop(r_id)
        """
        file_name = '{}.pickle'.format('test_kal_td') 
        traj_pickle_path = os.path.join('D:\Test',file_name)
        with open(traj_pickle_path, 'wb') as handle:
            pickle.dump(self.Off_tracking_pool, handle, protocol=pickle.HIGHEST_PROTOCOL)
        """        
        


            
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
                "win_size": [7, 13], 
                "eps": 1.5,
                "min_samples": 5,
                "bck_update_frame":500,
                "N":20,
                "d_thred":0.056,
                "bck_n" : 3,
                "bck_radius":0.9,
                "missing_thred":10,
                "if_save_pcd" : False,
                }


    output_file_path = r'D:\LiDAR_Data\MidTown\Martin'
    input_file_path = r'D:\LiDAR_Data\MidTown\Martin\2022-1-20-11-25-37R.pcap'
    mot = MOT(input_file_path,output_file_path,**params)
    mot.initialization()
    mot.mot_tracking(mot.frame_gen)
