import os

import imageio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from filterpy.stats import multivariate_gaussian
from matplotlib import cm
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

from DetectedObject import TrackingObject
from FrameGen import FrameGen
from Utils import get_color


class MultiTrackingSystem():
    def __init__(self,iter_n,tolerance,x_lim = [-80, 70],y_lim = [-20, 60],gen_fig = False):
        self.tracking_list = {}
        self.out_of_tracking_list = {}
        self.frame_gen = 0
        self.iter_n = iter_n
        self.tolerance = tolerance
        self.post_tracking_ind = 0
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.cur_frame = 0
        self.gen_fig = gen_fig
        
    def fit_dbgen(self,frames_name,eps,min_samples):
        self.frame_gen = FrameGen(frames_name,eps,min_samples).frame_generator()
        next_frame = next(self.frame_gen)
        for key in next_frame.keys():
            detected_obj = next_frame[key]
            self.tracking_list[self.post_tracking_ind] = TrackingObject(detected_obj.position,detected_obj.point_cloud,detected_obj.bounding_box)
            self.post_tracking_ind += 1
        if 'Gifs' not in os.listdir(r'./'):
            os.makedirs(r'./Gifs')
        self.visualization()


    def get_failed_new_ind(self,next_frame):
        center_probabilities_matrix = np.zeros(shape = (len(self.tracking_list.keys()),len(next_frame.keys())))
        for i,tracking_key in enumerate(self.tracking_list.keys()):
            self.tracking_list[tracking_key].tracker.predict() # tracker
            cov = np.array([[self.tracking_list[tracking_key].tracker.P_prior[0,0],self.tracking_list[tracking_key].tracker.P_prior[0,2]],
                        [self.tracking_list[tracking_key].tracker.P_prior[2,0],self.tracking_list[tracking_key].tracker.P_prior[2,2]]])
            mean = self.tracking_list[tracking_key].tracker.x_prior[[0,2]].T
            for j,next_key in enumerate(next_frame.keys()):
                center_probabilities_matrix[i,j] = multivariate_gaussian(next_frame[next_key].position,mu = mean, cov = cov)
        cur_tracking_ind, next_detection_ind = linear_sum_assignment(-center_probabilities_matrix)#*****
        """
        cur_tracking_ind: the relative position in the current tracking object list who can connect to a next trakcing object
        next_detection_ind: the relative position in the next frame object list who is selected by a current tracking object
        """
        failed_tracked_ind = np.setdiff1d(np.arange(len(self.tracking_list.keys())),cur_tracking_ind) # in the relative position who failed to tracking 
        new_detection_ind = np.setdiff1d(np.arange(len(next_frame.keys())),next_detection_ind)
        return failed_tracked_ind, new_detection_ind,cur_tracking_ind,next_detection_ind

    def drop_tracking(self,failed_key): # drop failed tracking object out of tracking list
        self.out_of_tracking_list[failed_key] = self.tracking_list[failed_key]
        _ = self.tracking_list.pop(failed_key)

    def track_next_frame(self):
        next_frame = next(self.frame_gen)
        tracking_obj_keys = list(self.tracking_list.keys())
        next_frame_obj_keys = list(next_frame.keys())
        failed_tracked_ind,new_detection_ind,cur_tracking_ind,next_detection_ind = self.get_failed_new_ind(next_frame)

        """
        Processing failed tracking
        """
        for i in failed_tracked_ind: 
            failed_key = tracking_obj_keys[i]
            if self.tracking_list[failed_key].failed_tracking_counting == self.tolerance:
                self.drop_tracking(failed_key)
            else:
                self.tracking_list[failed_key].tracker.predict()
                next_x,next_y = self.tracking_list[failed_key].tracker.x_prior[0][0],self.tracking_list[failed_key].tracker.x_prior[2][0]
                if (next_x<self.x_lim[0])|(next_x>self.x_lim[1])|(next_y>self.y_lim[1])|(next_y<self.y_lim[0]):
                    self.drop_tracking(failed_key)
                else:
                    self.tracking_list[failed_key].failed_tracking_counting+=1 # failed time +=1
                    self.tracking_list[failed_key].point_clouds.append(-1)# point cloud append -1
                    self.tracking_list[failed_key].estimated_centers.append(np.array([next_x,next_y])) # append pred x,y using only pred
                    self.tracking_list[failed_key].detected_centers.append(-1) # detection center
                    self.tracking_list[failed_key].bounding_boxes.append(-1) # bounding box
        """
        Processing new detection
        """
        for i in new_detection_ind:
            new_key = next_frame_obj_keys[i]
            detected_obj = next_frame[new_key]
            self.tracking_list[self.post_tracking_ind] = TrackingObject(detected_obj.position,detected_obj.point_cloud,detected_obj.bounding_box)
            self.post_tracking_ind += 1
        """
        Update tracking
        """
        for n,i in enumerate(cur_tracking_ind):
            next_key = next_detection_ind[n]
            cur_key = tracking_obj_keys[i]
            next_detection_position = next_frame[next_key].position
            cur_position = np.array([self.tracking_list[cur_key].tracker.x[0][0],self.tracking_list[cur_key].tracker.x[2][0]])

            if np.sqrt(np.sum((next_detection_position-cur_position)**2))>6:
                if (self.tracking_list[cur_key].failed_tracking_counting >= self.tolerance):
                    self.drop_tracking(cur_key)
                else:
                    self.tracking_list[cur_key].failed_tracking_counting +=1 # failed time +=1
                    self.tracking_list[cur_key].point_clouds.append(-1)# point cloud append -1
                    self.tracking_list[cur_key].tracker.predict()
                    next_x,next_y = self.tracking_list[cur_key].tracker.x_prior[0][0],self.tracking_list[cur_key].tracker.x_prior[2][0]
                    self.tracking_list[cur_key].estimated_centers.append(np.array([next_x,next_y])) # append pred x,y using only pred
                    self.tracking_list[cur_key].detected_centers.append(-1) # detection center
                    self.tracking_list[cur_key].bounding_boxes.append(-1) # bounding box
            else:
                self.tracking_list[cur_key].tracker.update(next_detection_position) # update
                self.tracking_list[cur_key].failed_tracking_counting = 0 # clear the failed times
                next_x,next_y = self.tracking_list[cur_key].tracker.x_post[0][0],self.tracking_list[cur_key].tracker.x_post[2][0]
                self.tracking_list[cur_key].estimated_centers.append(np.array([next_x,next_y]))
                self.tracking_list[cur_key].detected_centers.append(next_detection_position)
                self.tracking_list[cur_key].bounding_boxes.append(next_frame[next_key].bounding_box)
                self.tracking_list[cur_key].point_clouds.append(next_frame[next_key].point_cloud)
        
    
    def batch_tracking(self):
        for i in tqdm(range(self.iter_n-1)):
            self.cur_frame = i+1
            if self.gen_fig:
                self.visualization()
            self.track_next_frame()
        for key in self.tracking_list.keys():
            self.out_of_tracking_list[key] = self.tracking_list[key]
        self.tracking_list.clear()
            
    def visualization(self):
        plt.figure(figsize=(20,int(20*((self.y_lim[1]-self.y_lim[0])/(self.x_lim[1]-self.x_lim[0])))))
        plt.ylim(self.y_lim[0],self.y_lim[1]) # figure height is 80
        plt.xlim(self.x_lim[0],self.x_lim[1]) # length is 150
        plt.text(self.x_lim[0],self.y_lim[1],'{}'.format(self.cur_frame),fontsize = 25, c = 'blue')
        for key in self.tracking_list.keys():
            c = get_color(key)
            if type(self.tracking_list[key].detected_centers[-1]) == int:
                plt.scatter(self.tracking_list[key].estimated_centers[-1][0],self.tracking_list[key].estimated_centers[-1][1],s = 3, marker='o',c = 'g')
                plt.text(self.tracking_list[key].estimated_centers[-1][0],self.tracking_list[key].estimated_centers[-1][1],'{}e'.format(key),fontsize = 20,c = 'g') # estimate center
            else:
                plt.scatter(self.tracking_list[key].point_clouds[-1][:,0],self.tracking_list[key].point_clouds[-1][:,1],s = 3, marker = 'x',color = c)
                plt.plot(self.tracking_list[key].bounding_boxes[-1][:,0],self.tracking_list[key].bounding_boxes[-1][:,1],c = 'r') # box
                plt.plot(self.tracking_list[key].bounding_boxes[-1][[0,-1],0],self.tracking_list[key].bounding_boxes[-1][[0,-1],1],c = 'r')
                plt.text(self.tracking_list[key].detected_centers[-1][0],self.tracking_list[key].detected_centers[-1][1],'{}d'.format(key),fontsize = 20) # detection center
                plt.text(self.tracking_list[key].estimated_centers[-1][0],self.tracking_list[key].estimated_centers[-1][1],'{}e'.format(key),fontsize = 20) # e
        plt.savefig('./Gifs/{}.png'.format(self.cur_frame))
        plt.close()

    def svae_gif(self):
        images = []
        filenames = ['./Gifs/{}.png'.format(i) for i in range(self.iter_n)]
        for filename in filenames:
            images.append(imageio.imread(filename))
        imageio.mimsave('./tracking_result.gif', images)
        print('Gif successfully saved')
    
                
            
if __name__ == "__main__":
    folder = r'./frames/2019-8-27-7-0-0-BF1(0-18000frames)/{}'
    os.chdir(r'/Users/czhui960/Documents/Lidar/to ZHIHUI/US 395')
    frames_name = os.listdir(r'./frames/2019-8-27-7-0-0-BF1(0-18000frames)/')
    frames_name.sort(key = lambda x : x.split(' ')[2][:-5])
    test = MultiTrackingSystem(iter_n = 10, tolerance = 3)
    test.fit_dbgen(folder, frames_name, 2.0, 10)
    test.batch_tracking()
    # test.svae_gif()
