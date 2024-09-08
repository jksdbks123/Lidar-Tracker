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
        Step.1 -> Seperate output chunks including Td_map, Index_map, Foreground_map
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
        self.Foreground_map = None # A mask  indicating those pixels are required to be clustered
        self.Labeling_map_template = -1*np.ones(Td_map_szie,dtype = np.int16)
        self.Index_map = None # an intermediate variable
        self.Height_fringe_offset_fore = np.full((self.Height_fringe,Td_map_szie[1] + 2 * self.Width_fringe),False) 
        self.Height_fringe_offset_td = np.full((self.Height_fringe,Td_map_szie[1] + 2 * self.Width_fringe),200, dtype=np.float16) 
        self.Heigh_fringe_offset_index = np.full((self.Height_fringe,Td_map_szie[1] + 2 * self.Width_fringe),-1,dtype = np.int16) 

        
        
    def fit_predict(self,Td_map,Foreground_map):
        
        self.Td_map = Td_map
        self.Foreground_map = Foreground_map

        rows,cols = np.where(Foreground_map)
        indices = np.arange(len(rows),dtype = np.int64)
        self.Index_map = -1 * np.ones(shape = Foreground_map.shape,dtype=np.int64)
        self.Index_map[rows,cols] = indices # A map with the index of foreground point 
        
        # Horizontal padding
        
        Foreground_map_offset = np.concatenate([np.full((self.Foreground_map.shape[0],self.Width_fringe),False),
                                                self.Foreground_map,
                                                np.full((self.Foreground_map.shape[0],self.Width_fringe),False)],axis = 1)
        Index_map_offset = np.concatenate([np.full((self.Index_map.shape[0],self.Width_fringe),-1,dtype = np.int16),
                                        self.Index_map,
                                        np.full((self.Index_map.shape[0],self.Width_fringe),-1,dtype = np.int16)],axis = 1)
        Td_map_offset = np.concatenate([np.full((self.Td_map.shape[0],self.Width_fringe),200,dtype=np.float16),
                                        self.Td_map,
                                        np.full((self.Td_map.shape[0],self.Width_fringe),200,dtype=np.float16)],axis = 1)

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
        # self.Sub_foremap,self.Sub_indmap,self.Sub_tdmap = Sub_foremap,Sub_indmap,Sub_tdmap
        valid_windows = Sub_foremap[:,self.Height_fringe ,self.Width_fringe] 
        # self.valid_windows = valid_windows
        Sub_indmap,Sub_foremap,Sub_tdmap = Sub_indmap[valid_windows],Sub_foremap[valid_windows],Sub_tdmap[valid_windows]
        
        # ***key step
        center_td_dist = Sub_tdmap[:,self.Height_fringe,self.Width_fringe]
        temp = ((np.abs((Sub_tdmap - center_td_dist.reshape(-1,1,1))) < self.eps) & Sub_foremap)
        neighborhoods = np.array([Sub_indmap[i][temp[i]] for i in range(len(temp))],dtype = 'O')
        n_neighbors = np.array([len(neighbor) for neighbor in neighborhoods])
        Labels = np.full(len(n_neighbors), -1, dtype=np.intp)
        core_samples = np.asarray(n_neighbors >= self.min_samples,dtype=np.uint8)
        
        try:
            dbscan_inner(core_samples, neighborhoods, Labels)
        except:
            pass
        Labeling_map = self.Labeling_map_template.copy()
        # Labels = Labels.astype(int)
        Labeling_map[rows,cols] = Labels
        # self.Labeling_map = Labeling_map
        
        return Labeling_map

# if __name__ == '__main__':
#     # Test the Raster_DBSCAN
#     Td_map = np.random.rand(32,1800)
#     Foreground_map = np.random.rand(32,1800) > 0.5
#     db = Raster_DBSCAN(window_size=(5,5),eps = 0.1,min_samples=5,Td_map_szie=(32,1800))
#     Labeling_map = db.fit_predict(Td_map,Foreground_map)
#     print(Labeling_map)
#     pass