from sklearn.neighbors import NearestNeighbors
from sklearn.cluster._dbscan_inner import dbscan_inner
import numpy as np
class AdaptiveDBSCAN():
    def __init__(self,beta,min_sample_1,min_sample_2,min_sample_3):
        self.X = 0
        self.beta = beta
        self.corr = np.sin((np.pi * (0.2)/180)/2) * 2
        self.min_sample_1 = min_sample_1
        self.min_sample_2 = min_sample_2
        self.min_sample_3 = min_sample_3
    
    def fit_predict(self,X):
        self.X = X
        neighborhoods = []
        for i in range(len(self.X)):
            radius = self.beta * self.corr * self.X[i,2] 
            neighbors_model = NearestNeighbors(radius=radius,leaf_size=30,p = 2)
            neighbors_model.fit(self.X[:,:2])
            neighborhood = neighbors_model.radius_neighbors(self.X[i,:2].reshape(1,-1),return_distance=False)
            neighborhoods.append(neighborhood[0])
        neighborhoods = np.array(neighborhoods)
        n_neighbors = np.array([len(n) for n in neighborhoods])
        min_points = self.min_sample_1 * self.X[:,2]**(self.min_sample_2) + self.min_sample_3
        min_points[min_points<3] = 3
        core_samples = np.asarray(n_neighbors >= 5,dtype=np.uint8)
        labels = np.full(self.X.shape[0], -1, dtype=np.intp)
        dbscan_inner(core_samples, neighborhoods, labels)
        return labels 