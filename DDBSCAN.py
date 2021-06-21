import numpy as np
from sklearn.cluster._dbscan_inner import dbscan_inner
from numpy.lib.stride_tricks import sliding_window_view

class Raster_DBSCAN():
    def __init__(self,window_size,eps,min_samples,Td_map_szie):
        """
        This method applys the vectorization calculation, 
        where the TD map is chunked as multiple parts and corresponding neighbors in each center of chunk are counted and recorded.
        The calculated neighbors are stored in the neighborhood list which is then treated as an input for the dbscan_inner() so as to
        utilize the core CPP API from the sklearn. 

        *** 
        Step.1 -> Chunking output Subchunks including Td_map, Index_map, Foreground_map
        Step.2 -> Neigborhoods calculation
        Step.3 -> dbscan_inner()

        """
        self.window_size = window_size #(height,width)
        self.Td_map_size = Td_map_szie
        self.eps = eps
        self.min_samples = min_samples
        self.Height_fringe = int(self.window_size[0]/2) 
        self.Width_fringe = int(self.window_size[1]/2) # azimuth
        self.Td_map = None #Two-Dimentional Map   
        self.Labeling_map = None #
        self.Foreground_map = None # A mask  indicating those pixels are required to be clustered
        self.Index_map = None # an intermediate variable
        self.Height_fringe_offset_fore = np.full((self.Height_fringe,Td_map_szie[1] + 2 * self.Width_fringe),False) 
        self.Height_fringe_offset_td = np.full((self.Height_fringe,Td_map_szie[1] + 2 * self.Width_fringe),200) 
        self.Heigh_fringe_offset_index = np.full((self.Height_fringe,Td_map_szie[1] + 2 * self.Width_fringe),-1,dtype = np.int64) 
        
        
    def fit_predict(self,Td_map,Foreground_map):
        
        self.Td_map = Td_map
        self.Foreground_map = Foreground_map

        rows,cols = np.where(Foreground_map)
        indices = np.arange(len(rows),dtype = np.int64)
        self.Index_map = -1*np.ones(shape = Foreground_map.shape,dtype=np.int64)
        self.Index_map[rows,cols] = indices 
        
        # Horizontal padding 
        Foreground_map_offset = np.concatenate([self.Foreground_map[:,-self.Width_fringe:],
                                                self.Foreground_map,
                                                self.Foreground_map[:,:self.Width_fringe]],axis = 1)
        Index_map_offset = np.concatenate([self.Index_map[:,-self.Width_fringe:],
                                        self.Index_map,
                                        self.Index_map[:,:self.Width_fringe]],axis = 1)
        Td_map_offset = np.concatenate([self.Td_map[:,-self.Width_fringe:],
                                        self.Td_map,
                                        self.Td_map[:,:self.Width_fringe]],axis = 1)
        # Vertical padding 
        Foreground_map_offset = np.concatenate([self.Height_fringe_offset_fore,
                                                Foreground_map_offset,
                                            self.Height_fringe_offset_fore])
        Index_map_offset = np.concatenate([self.Heigh_fringe_offset_index,
                                        Index_map_offset,
                                        self.Heigh_fringe_offset_index])
        Td_map_offset = np.concatenate([self.Height_fringe_offset_td,
                                    Td_map_offset,
                                    self.Height_fringe_offset_td])
        
        Sub_indmap = sliding_window_view(Index_map_offset,self.window_size).reshape(-1,self.window_size[0],self.window_size[1])
        Sub_tdmap = sliding_window_view(Td_map_offset,self.window_size).reshape(-1,self.window_size[0],self.window_size[1])
        Sub_foremap = sliding_window_view(Foreground_map_offset,self.window_size).reshape(-1,self.window_size[0],self.window_size[1])
        # Window inds that are valid as Foregound, and only 
        valid_windows = Sub_foremap[:,self.Height_fringe,self.Width_fringe] 
        Sub_indmap,Sub_foremap,Sub_tdmap = Sub_indmap[valid_windows],Sub_foremap[valid_windows],Sub_tdmap[valid_windows]
        
        # ***key step
        center_td_dist = Sub_tdmap[:,self.Height_fringe,self.Width_fringe]
        temp = ((np.abs((Sub_tdmap - center_td_dist.reshape(-1,1,1))) < self.eps) & Sub_foremap)
        neighborhoods = np.array([Sub_indmap[i][temp[i]] for i in range(len(temp))],dtype = 'O')
        n_neighbors = np.array([len(neighbor) for neighbor in neighborhoods])
        Labels = np.full(len(n_neighbors), -1, dtype=np.intp)
        core_samples = np.asarray(n_neighbors >= self.min_samples,dtype=np.uint8)
        dbscan_inner(core_samples, neighborhoods, Labels)

        Labeling_map = -1*np.ones_like(Foreground_map)
        Labeling_map[rows,cols] = Labels
        self.Labeling_map = Labeling_map
        
        return Labeling_map
