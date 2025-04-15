import numpy as np
import os
from collections import defaultdict
from scipy.interpolate import interp1d
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from filterpy.common import Q_discrete_white_noise

def get_trajs_from_Kalman_out(Labels,center = False, max_prediction_count = 25):
    n_lane_cells, n_frames = Labels.shape
    # Initialize storage for active tracks and completed trajectories
    active_tracks = {}  # {track_id: {'filter': filter, 'last_update': t, 'pred_count': n, 'positions': [(t, pos)]}}
    trajectories = {}   # {track_id: [(t, pos)]}
    next_track_id = 0   # Counter for assigning unique track IDs
    cur_t = 0
    while True:
        Labels_t = Labels[:, cur_t]
        unique_id, xs, counts = np.unique(Labels_t, return_index=True, return_counts=True)
        # Skip background (-1) if present
        if unique_id[0] == -1:
            unique_id = unique_id[1:]
            xs = xs[1:]
            counts = counts[1:]
        if len(unique_id) > 0:
            if center:
                positions = xs + 0.5 * counts
            else:
                positions = xs
            current_detections = list(zip(unique_id, positions))
            for det_id, det_pos in current_detections:
                # Initialize new Kalman filter
                kf = KalmanFilter(dim_x=2, dim_z=1)
                kf.x = np.array([[det_pos],  # position
                                    [0.]])      # velocity
                kf.F = np.array([[1., 1.],
                                    [0., 1.]])
                kf.H = np.array([[1., 0.]])
                kf.P *= 1000.
                kf.R = 2000
                kf.Q = 0.5
                
                # Create new track
                track_id = next_track_id
                next_track_id += 1
                
                active_tracks[track_id] = {
                    'filter': kf,
                    'last_update': cur_t,
                    'pred_count': 0,
                    'positions': [(cur_t, det_pos)]
                }
        
            break
        cur_t += 1


    for t in range(cur_t,n_frames):
        # print(f'Processing frame {t}')
        # Extract vehicle positions in current frame
        Labels_t = Labels[:, t]
        unique_id, xs, counts = np.unique(Labels_t, return_index=True, return_counts=True)
        # Skip background (-1) if present
        if unique_id[0] == -1:
            unique_id = unique_id[1:]
            xs = xs[1:]
            counts = counts[1:]
        # Calculate vehicle positions
        if center:
            positions = xs + 0.5 * counts
        else:
            positions = xs
        # Store current detections for association
        current_detections = list(zip(unique_id, positions))
        for track_id, track_data in list(active_tracks.items()):
            kf = track_data['filter']
            kf.predict()
            # Extract predicted position (x[0])
            predicted_pos = float(kf.x[0, 0])
            # Update track data with prediction
            track_data['predicted_pos'] = predicted_pos
        # Step 2: Associate detections with existing tracks using Hungarian algorithm
        if len(active_tracks) > 0 and len(current_detections) > 0:
            # Create cost matrix
            cost_matrix = np.zeros((len(active_tracks), len(current_detections)))
            for i, (track_id, track_data) in enumerate(active_tracks.items()):
                for j, (det_id, det_pos) in enumerate(current_detections):
                    # Calculate distance between predicted position and detection
                    distance = abs(track_data['predicted_pos'] - det_pos)
                    
                    # Apply a maximum threshold for association (e.g., 5 lane cells)
                    if distance > 5.0:
                        cost_matrix[i, j] = 1000  # Large cost for impossible associations
                    else:
                        cost_matrix[i, j] = distance
            # Apply Hungarian algorithm
            track_indices, detection_indices = linear_sum_assignment(cost_matrix)
            # Process associations
            associated_tracks = set()
            associated_detections = set()
            for i, j in zip(track_indices, detection_indices):
                # Only associate if the cost is below threshold
                if cost_matrix[i, j] < 1000:
                    # Get track and detection
                    track_id = list(active_tracks.keys())[i]
                    det_id, det_pos = current_detections[j]
                    
                    # Update Kalman filter
                    track_data = active_tracks[track_id]
                    kf = track_data['filter']
                    kf.update(det_pos)
                    
                    # Update track data
                    track_data['last_update'] = t
                    track_data['pred_count'] = 0  # Reset consecutive prediction count
                    track_data['positions'].append((t, det_pos))
                    
                    associated_tracks.add(track_id)
                    associated_detections.add(det_id)
        else:
            associated_tracks = set()
            associated_detections = set()
        # Step 3: Create new tracks for unassociated detections
        for det_id, det_pos in current_detections:
            if det_id not in associated_detections:
                # Initialize new Kalman filter
                kf = KalmanFilter(dim_x=2, dim_z=1)
                kf.x = np.array([[det_pos],  # position
                                    [0.]])      # velocity
                kf.F = np.array([[1., 1.],
                                    [0., 1.]])
                kf.H = np.array([[1., 0.]])
                kf.P *= 1000.
                kf.R = 2000
                kf.Q = 0.5
                
                # Create new track
                track_id = next_track_id
                next_track_id += 1
                
                active_tracks[track_id] = {
                    'filter': kf,
                    'last_update': t,
                    'pred_count': 0,
                    'positions': [(t, det_pos)]
                }
        # Step 4: Update unassociated tracks with prediction only
        for track_id, track_data in list(active_tracks.items()):
            if track_id not in associated_tracks:
                # Track was not associated with a detection
                track_data['pred_count'] += 1
                
                # If consecutive predictions exceed threshold, terminate track
                if track_data['pred_count'] > max_prediction_count:
                    # Move to completed trajectories
                    trajectories[track_id] = track_data['positions']
                    del active_tracks[track_id]
                else:
                    # Continue track with prediction only
                    kf = track_data['filter']
                    kf.predict()
                    # Extract predicted position (x[0])
                    predicted_pos = float(kf.x[0, 0])
                    track_data['positions'].append((t, predicted_pos))
    # Add remaining active tracks to trajectories
    for track_id, track_data in active_tracks.items():
        trajectories[track_id] = track_data['positions']
    trajectories_ = []
    for track_id, positions in trajectories.items():
        # Convert positions to numpy array
        positions = np.array(positions)
        # Append to list of trajectories
        trajectories_.append(positions)
    return trajectories_