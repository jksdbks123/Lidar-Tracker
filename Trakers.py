from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import KalmanFilter
from filterpy.stats import plot_covariance_ellipse
import numpy as np

def kalman_tracker(ini_x,ini_y):
    dt = 1.0 # time step
    R_std = 4
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    tracker.F = np.array([[1, dt, 0,  0],
                        [0,  1, 0,  0],
                        [0,  0, 1, dt],
                        [0,  0, 0,  1]])
    Q = np.array([
        [1,0,0,0],
        [0,1.5,0,0],
        [0,0,1,0],
        [0,0,0,1.5]
    ])
    tracker.Q = Q
    tracker.H = np.array([[1, 0, 0, 0],
                        [0, 0, 1, 0]])
    tracker.R = np.eye(2) * R_std**2
    tracker.x = np.array([[ini_x, 0, ini_y, 0]]).T
    tracker.P = np.array([
        [16,0,0,0],
        [0,9,0,0],
        [0,0,16,0],
        [0,0,0,9]
    ])
    return tracker