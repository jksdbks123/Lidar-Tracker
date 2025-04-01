import numpy as np
import os
from collections import defaultdict
from scipy.interpolate import interp1d
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from filterpy.common import Q_discrete_white_noise
def get_pcd_colored(Td_map,Labeling_map,vertical_limits):

    Xs = []
    Ys = []
    Labels = []
    for i in range(vertical_limits[0],vertical_limits[1]):
        longitudes = theta[i]*np.pi / 180
        latitudes = azimuths * np.pi / 180 
        hypotenuses = Td_map[i] * np.cos(longitudes)
        X = hypotenuses * np.sin(latitudes)
        Y = hypotenuses * np.cos(latitudes)
        Xs.append(X)
        Ys.append(Y)
        Labels.append(Labeling_map[i])

    Xs = np.concatenate(Xs)
    Ys = np.concatenate(Ys)
    Labels = np.concatenate(Labels)
    XYZ = np.c_[Xs,Ys]
    Valid_ind = (XYZ[:,0] != 0)&(XYZ[:,1] != 0)
    Labels = Labels[Valid_ind]
    XYZ = XYZ[Valid_ind]

    return XYZ,Labels  
def calc_timing_offsets():
    timing_offsets = np.zeros((32,12))  # Init matrix
    # constants
    full_firing_cycle = 55.296  # μs
    single_firing = 2.304  # μs
    # compute timing offsets
    for x in range(12):
        for y in range(32):
            dataBlockIndex = (x * 2) + int((y / 16))
            dataPointIndex = y % 16
            timing_offsets[y][x] = (full_firing_cycle * dataBlockIndex) +(single_firing * dataPointIndex)

    return np.array(timing_offsets).T

Data_order = np.array([[-25,1.4],[-1,-4.2],[-1.667,1.4],[-15.639,-1.4],
                            [-11.31,1.4],[0,-1.4],[-0.667,4.2],[-8.843,-1.4],
                            [-7.254,1.4],[0.333,-4.2],[-0.333,1.4],[-6.148,-1.4],
                            [-5.333,4.2],[1.333,-1.4],[0.667,4.2],[-4,-1.4],
                            [-4.667,1.4],[1.667,-4.2],[1,1.4],[-3.667,-4.2],
                            [-3.333,4.2],[3.333,-1.4],[2.333,1.4],[-2.667,-1.4],
                            [-3,1.4],[7,-1.4],[4.667,1.4],[-2.333,-4.2],
                            [-2,4.2],[15,-1.4],[10.333,1.4],[-1.333,-1.4]
                            ])
laser_id = np.full((32,12),np.arange(32).reshape(-1,1).astype('int'))
timing_offset = calc_timing_offsets()
omega = Data_order[:,0]
theta = np.sort(omega)
azimuths = np.arange(0,360,0.2)
arg_omega = np.argsort(omega)

# connection_pairs = [(0,1),(1,2),(2,3),(5,6)]
# identify chains in the connection_pairs and return the chain list
def identify_chains(connection_pairs):
    chains = []
    for pair in connection_pairs:
        if len(chains) == 0:
            chains.append(list(pair))
        else:
            pair_inserted = False
            for chain in chains:
                if pair[0] in chain:
                    chain.append(pair[1])
                    pair_inserted = True
                    break
                elif pair[1] in chain:
                    chain.append(pair[0])
                    pair_inserted = True
                    break
            if not pair_inserted:
                chains.append(list(pair))
    return chains
# connection_pairs_ = identify_chains(connection_pairs)
def get_traj_labels(traj_dir,trajectory,time_space_diagram):
    # read labels in txt file in keypoint format for yolov8 and extract the keypoint coordinates
    lines = []
    with open(os.path.join(traj_dir,trajectory),'r') as f:
        for line in f:
            line = line.strip().split()
            line = [float(i) for i in line]
            lines.append(line)
    lines = np.array(lines)
    trajs_label = []
    # from index 8 of each row is the keypoint coordinates: something, x,y , something, x,y ...
    traj_type_label = []
    for line in lines:
        coords = []
        for i in range(5,len(line),3):
            coords.append((line[i],line[i+1]))
        coords = np.array(coords)
        coords[:,0] *= time_space_diagram.shape[1]
        coords[:,1] *= time_space_diagram.shape[0]
        traj = coords[coords[:,0].argsort()]
        trajs_label.append(traj)
        traj_type_label.append(line[0])
    return traj_type_label,trajs_label

def get_trajctories_dict_from_Label_map(Labels,time_span,center = True):
    trajectories_output = defaultdict(list)
    for t in range(time_span):
        Labels_t = Labels[:,t]
        unique_id, xs, counts = np.unique(Labels_t,return_index=True,return_counts=True)
        if center:
            center_locations = xs + 0.5*counts
        else:
            center_locations = xs
        if unique_id[0] == -1:
            unique_id = unique_id[1:]
            center_locations = center_locations[1:]
        for i, label in enumerate(unique_id):
            trajectories_output[label].append((t,center_locations[i]))
    for key in trajectories_output.keys():
        traj = trajectories_output[key]
        trajectories_output[key] = np.array(traj)
    return trajectories_output

def get_trajs_from_LSTM_out(time_space_diagram,db,conf_thred = 0.35,time_span = 100, lane_unit = 200,center = True):
    pred_trajectories = time_space_diagram > conf_thred
    Labels = db.fit_predict(pred_trajectories, pred_trajectories)
    num_lane_units, time_span = time_space_diagram.shape
    trajectories_output = get_trajctories_dict_from_Label_map(Labels,time_span,center)
    # convert to list
    trajectories_output_ = []
    for key in trajectories_output.keys():
        traj = trajectories_output[key]
        trajectories_output_.append(traj)
    return trajectories_output_

def get_trajs_from_Kalman_out(time_space_diagram_input,db,time_span = 100):
    Labels = db.fit_predict(time_space_diagram_input> 0, time_space_diagram_input > 0)
    trajs_pred = get_trajctories_dict_from_Label_map(Labels,time_span,center = False)
    future_time_span = 25 # for kalmann filter
    connection_pairs = []
    for key in trajs_pred.keys():
        traj = np.array(trajs_pred[key])
        xs = traj[:,1]
        ts = traj[:,0]
        # interpolate the trajectory
        interpfunc = interp1d(ts, xs, kind='linear')
        ts = np.arange(ts[0],ts[-1]+1)
        xs = interpfunc(ts)
        f = KalmanFilter(dim_x=2, dim_z=1)
        f.x = np.array([[xs[0]],  # position
                        [0.]]) # velocity
        f.F = np.array([[1.,1.],
                        [0.,1.]])
        f.H = np.array([[1.,0.]])
        f.P *= 1000.
        f.R = 2000
        f.Q = 0.5
        kalman_traj = [[ts[0],xs[0]]]
        for i,t in enumerate(ts):
            f.predict()
            f.update(xs[i])
            kalman_traj.append([t,f.x[0][0]])

        find_connection_flag = False
        for t in range(ts[-1],ts[-1]+future_time_span):
            f.predict()
            for key_ in trajs_pred.keys():
                if key_ == key:
                    continue
                x_residual = np.abs(f.x[0][0] - trajs_pred[key_][0,1])
                t_residual =  trajs_pred[key_][0,0] - t 
                if x_residual < 2 and t_residual == 0:
                    connection_pairs.append((key,key_))
                    find_connection_flag = True
                    break
            if find_connection_flag:
                break
            kalman_traj.append([t,f.x[0][0]])
    trajs_pred_ = trajs_pred.copy()
    connection_pairs_ = identify_chains(connection_pairs)
    for chain in connection_pairs_:
        for i in range(1,len(chain)):
            try:
                trajs_pred_[chain[0]] = np.concatenate([trajs_pred_[chain[0]],trajs_pred_[chain[i]]],axis = 0)
                trajs_pred_.pop(chain[i])
            except:
                continue
    trajs_pred_ = [trajs_pred_[key] for key in trajs_pred_.keys()]

    return trajs_pred_

# evaluate 
speed_eva_window = 5
def get_traj_errors(trajs_label,trajs_pred,speed_eva_window = 5):
    speed_errors = []
    location_errors = []
    for i, traj_label in enumerate(trajs_label):
        for j, traj_pred in enumerate(trajs_pred.values()):
            if len(traj_pred) < speed_eva_window:
                continue
            # Extract time and x values
            t_pred, x_pred = traj_pred[:, 0], traj_pred[:, 1]
            # drop duplicate time points
            t_pred, indices = np.unique(t_pred, return_index=True)
            x_pred = x_pred[indices]
            t_label, x_label = traj_label[:, 0], traj_label[:, 1]
            # drop duplicate time points
            t_pred, indices = np.unique(t_label, return_index=True)
            x_label = x_label[indices]
            # Interpolate label trajectory to match pred trajectory time points
            interp_func = interp1d(t_label, x_label, fill_value="extrapolate")
            x_label_interp = interp_func(t_pred)
            
            # get the overlapping time span
            t_label_max,t_label_min = t_label[-1],t_label[0]
            t_pred_max,t_pred_min = t_pred[-1],t_pred[0]
            # if two trajectories have overlapping time span

            if t_pred_max < t_label_min or t_pred_min > t_label_max:
                continue
            overlap_start = max(t_pred_min,t_label_min)
            overlap_end = min(t_pred_max,t_label_max)
            x_label_interp_overlap = x_label_interp[(t_pred >= overlap_start) & (t_pred <= overlap_end)]
            x_pred_overlap = x_pred[(t_pred >= overlap_start) & (t_pred <= overlap_end)]
            # Calculate residuals
            residuals = x_pred_overlap - x_label_interp_overlap
            if len(residuals) == 0:
                continue
            location_error = np.mean(np.abs(residuals)) * 0.5 # meters
            
            if location_error < 1: # they are the same trajectory
      
                location_errors.append(location_error)            

                # calculate speed error
                speed_pred_curve = []
                speed_label_curve = []
                for k in range(len(residuals)-speed_eva_window):
                    speed_pred = (x_pred_overlap[k+speed_eva_window] - x_pred_overlap[k])*0.5 / (speed_eva_window/10)
                    speed_label = (x_label_interp_overlap[k+speed_eva_window] - x_label_interp_overlap[k])*0.5 / (speed_eva_window/10)
                    speed_pred_curve.append(speed_pred)
                    speed_label_curve.append(speed_label)
                speed_pred_curve = np.array(speed_pred_curve)
                speed_label_curve = np.array(speed_label_curve)
                if len(speed_pred_curve) == 0:
                    continue
                speed_error = np.mean(np.abs(speed_pred_curve - speed_label_curve)) # m/s
                if speed_error > 4:
                    print(location_error,speed_error, i,j)
                    plt.plot(x_label_interp_overlap,'b')
                    plt.plot(x_pred_overlap,'r')
                    plt.show()
                speed_errors.append(speed_error)
    return location_errors, speed_errors

def get_interpolated_traj(traj_label,traj_pred):
    t_pred, x_pred = traj_pred[:, 0], traj_pred[:, 1]
    # drop duplicate time points
    t_pred, indices = np.unique(t_pred, return_index=True)
    x_pred = x_pred[indices]
    t_label, x_label = traj_label[:, 0], traj_label[:, 1]
    # drop duplicate time points
    t_label, indices = np.unique(t_label, return_index=True)
    x_label = x_label[indices]
    # get the overlapping time span
    t_label_max,t_label_min = int(t_label.max()),int(t_label.min())
    t_pred_max,t_pred_min = int(t_pred.max()),int(t_pred.min())
   

    t_pred_ = np.arange(t_pred_min,t_pred_max+1)
    t_label_ = np.arange(t_label_min,t_label_max+1)
    # Interpolate label trajectory to match pred trajectory time points
    interp_func_label = interp1d(t_label, x_label, fill_value="extrapolate")
    interp_func_pred = interp1d(t_pred, x_pred, fill_value="extrapolate")
    x_label_interp = interp_func_label(t_pred_)
    x_pred_interp = interp_func_pred(t_label_)
    overlap_start = max(t_pred_min,t_label_min)
    overlap_end = min(t_pred_max,t_label_max)
    union_start = min(t_pred_min,t_label_min)
    union_end = max(t_pred_max,t_label_max)
    num_union = union_end - union_start + 1
    num_overlap = overlap_end - overlap_start + 1
    x_label_interp_overlap = x_label_interp[(t_pred_ >= overlap_start) & (t_pred_ <= overlap_end)]
    x_pred_interp_overlap = x_pred_interp[(t_label_ >= overlap_start) & (t_label_ <= overlap_end)]
    return t_pred_,t_label_,x_label_interp,x_pred_interp,x_label_interp_overlap,x_pred_interp_overlap,num_union,num_overlap,union_start,union_end,overlap_start,overlap_end

def get_TrajIoU(trajs_label,trajs_pred,x_error_threshold = 1):
    TrajIoUMatrix = np.zeros((len(trajs_label),len(trajs_pred)))
    TrajPortionMatrix = np.zeros((len(trajs_label),len(trajs_pred)))
    for i, traj_label in enumerate(trajs_label):
        for j, traj_pred in enumerate(trajs_pred):
            # Extract time and x values
            t_pred, x_pred = traj_pred[:, 0], traj_pred[:, 1]
            t_label, x_label = traj_label[:, 0], traj_label[:, 1]
            # get the overlapping time span
            t_label_max,t_label_min = int(t_label.max()),int(t_label.min())
            t_pred_max,t_pred_min = int(t_pred.max()),int(t_pred.min())
            # if two trajectories have no overlapping time span, skip
            if t_pred_max < t_label_min or t_pred_min > t_label_max:
                continue
            t_pred_,t_label_,x_label_interp,x_pred_interp,x_label_interp_overlap,x_pred_interp_overlap,num_union,num_overlap,union_start,union_end,overlap_start,overlap_end = get_interpolated_traj(traj_label,traj_pred)
            
            x_label_interp_overlap = x_label_interp[(t_pred_ >= overlap_start) & (t_pred_ <= overlap_end)]
            x_pred_interp_overlap = x_pred_interp[(t_label_ >= overlap_start) & (t_label_ <= overlap_end)]
            residuals = np.abs(x_label_interp_overlap - x_pred_interp_overlap)
            if len(residuals) == 0:
                continue
            TrajIoUMatrix[i,j] = (residuals < x_error_threshold).sum() / num_overlap
            # how long the pred trajectory is in the label trajectory
            TrajPortionMatrix[i,j] = (residuals < x_error_threshold).sum() / num_overlap

    return TrajIoUMatrix,TrajPortionMatrix

def get_accuracy_metric(Iou_threshold, TrajIoUMatrix, TrajPortionMatrix, trajs_pred):
    pred_index = TrajIoUMatrix.argmax(axis = 1)
    pred_IoU = TrajIoUMatrix.max(axis = 1)
    tp, fp, fn, id_switch, discontinuity = 0, 0, 0, 0, 0
    for i,j in enumerate(pred_index):
        # i is the index of label trajectory, j is the index of pred trajectory
        IoU = pred_IoU[i]
        if IoU > Iou_threshold:
            tp += 1
        else:
            fn += 1
    
    for i in range(TrajIoUMatrix.shape[0]):
        if (TrajIoUMatrix[i] > 0).sum() > 1:
            discontinuity += 1
    for j in range(TrajIoUMatrix.shape[1]):
        if (TrajIoUMatrix[:,j] > 0).sum() > 1:
            id_switch += 1
    for j in range(TrajIoUMatrix.shape[1]):
        if j not in pred_index:
            if trajs_pred[j][0,1] > 5:
                fp += 1

    return tp, fp, fn, id_switch, discontinuity

def get_speed_spatial_error(trajs_label,trajs_pred,pred_index,pred_IoU,Iou_threshold,speed_eva_window,traj_type_label = None):
    acceleration_errors = []
    speed_errors = []
    spatial_errors = []
    occ_acceleration_errors = []
    occ_speed_errors = []
    occ_spatial_errors = []
    

    for i,j in enumerate(pred_index):
        if pred_IoU[i] >= Iou_threshold and len(trajs_pred[j]) > speed_eva_window + 1:
   

            traj_label = trajs_label[i]
            traj_pred = trajs_pred[j]
            t_pred, x_pred = traj_pred[:, 0], traj_pred[:, 1]
            t_label, x_label = traj_label[:, 0], traj_label[:, 1]
        # get the overlapping time span
            t_label_max,t_label_min = int(t_label.max()),int(t_label.min())
            t_pred_max,t_pred_min = int(t_pred.max()),int(t_pred.min())
            if t_pred_max < t_label_min or t_pred_min > t_label_max:
                continue
            t_pred_,t_label_,x_label_interp,x_pred_interp,x_label_interp_overlap,x_pred_interp_overlap,num_union,num_overlap,union_start,union_end,overlap_start,overlap_end =  get_interpolated_traj(traj_label,traj_pred)
            sptial_error = x_pred_interp_overlap - x_label_interp_overlap
            if traj_type_label[i] == 1:
                occ_spatial_errors.append(sptial_error)
            else:
                spatial_errors.append(sptial_error)
        # calculate speed error
            speed_pred_curve = []
            acceleration_pred_curve = []
            speed_label_curve = []
            acceleration_label_curve = []
            for k in range(len(sptial_error)-speed_eva_window):
                speed_pred_k = (x_pred_interp_overlap[k+speed_eva_window] - x_pred_interp_overlap[k])*0.5 / (speed_eva_window/10)
                speed_label_k = (x_label_interp_overlap[k+speed_eva_window] - x_label_interp_overlap[k])*0.5 / (speed_eva_window/10)
                speed_pred_k_ = (x_pred_interp_overlap[k+speed_eva_window-1] - x_pred_interp_overlap[k-1])*0.5 / (speed_eva_window/10)
                speed_label_k_ = (x_label_interp_overlap[k+speed_eva_window-1] - x_label_interp_overlap[k-1])*0.5 / (speed_eva_window/10)
                acceleration_pred_k = speed_pred_k - speed_pred_k_
                acceleration_label_k = speed_label_k - speed_label_k_
                
                speed_pred_curve.append(speed_pred_k)
                speed_label_curve.append(speed_label_k)
                acceleration_pred_curve.append(acceleration_pred_k)
                acceleration_label_curve.append(acceleration_label_k)
            
            speed_pred_curve = np.array(speed_pred_curve)
            speed_label_curve = np.array(speed_label_curve)
            acceleration_pred_curve = np.array(acceleration_pred_curve)
            acceleration_label_curve = np.array(acceleration_label_curve)
            
            speed_error = speed_pred_curve - speed_label_curve
            acceleration_error = acceleration_pred_curve - acceleration_label_curve
            if traj_type_label[i] == 0:
                occ_speed_errors.append(speed_error)
                occ_acceleration_errors.append(acceleration_error)
            else:
                speed_errors.append(speed_error)
                acceleration_errors.append(acceleration_error)
            # speed_errors.append(speed_error)
    
    if len(speed_errors) == 0:
        speed_errors = np.array([])
    else:
        speed_errors = np.concatenate(speed_errors)
    if len(spatial_errors) == 0:
        spatial_errors = np.array([])
    else:
        spatial_errors = np.concatenate(spatial_errors)
    if len(occ_speed_errors) == 0:
        occ_speed_errors = np.array([])
    else:
        occ_speed_errors = np.concatenate(occ_speed_errors)
    if len(occ_spatial_errors) == 0:
        occ_spatial_errors = np.array([])
    else:
        occ_spatial_errors = np.concatenate(occ_spatial_errors)
    if len(acceleration_errors) == 0:
        acceleration_errors = np.array([])
    else:
        acceleration_errors = np.concatenate(acceleration_errors)
    if len(occ_acceleration_errors) == 0:
        occ_acceleration_errors = np.array([])
    else:
        occ_acceleration_errors = np.concatenate(occ_acceleration_errors)
    return speed_errors,spatial_errors,occ_speed_errors,occ_spatial_errors,acceleration_errors,occ_acceleration_errors
