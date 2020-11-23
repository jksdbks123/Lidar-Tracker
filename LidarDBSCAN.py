from sklearn.neighbors import NearestNeighbors
from sklearn.cluster._dbscan_inner import dbscan_inner
import numpy as np
class AdaptiveDBSCAN():
    def __init__(self,beta,min_sample_1,min_sample_2,min_sample_3,delta = 0.15,c_min = True):
        self.X = 0
        self.beta = beta
        self.corr = np.sin((np.pi * (0.2)/180)/2) * 2
        self.min_sample_1 = min_sample_1
        self.min_sample_2 = min_sample_2
        self.min_sample_3 = min_sample_3
        self.c_min = c_min
        self.delta = delta
    def fit_predict(self,X):
        self.X = X
        neighborhoods = []
        for i in range(len(self.X)):
            if self.X[i,2]<10:
                radius = 0.9
            else:
                radius = self.beta * self.corr * self.X[i,2] + self.delta
            nnds = np.sqrt(np.sum((self.X[i,:2] - self.X[:,:2])**2,axis=1))
            neighborhoods.append(np.where(nnds<radius)[0])
        neighborhoods = np.array(neighborhoods)
        n_neighbors = np.array([len(n) for n in neighborhoods])
        if self.c_min == True:
            min_points = 5
        else:
            min_points = self.min_sample_1 * self.X[:,2]**(self.min_sample_2) + self.min_sample_3
            min_points[min_points<3] = 3
        core_samples = np.asarray(n_neighbors >= min_points,dtype=np.uint8)
        labels = np.full(self.X.shape[0], -1, dtype=np.intp)
        dbscan_inner(core_samples, neighborhoods, labels)
        return labels 