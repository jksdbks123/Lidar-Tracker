import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import random
import cv2
import time
import os
# times new roman font for matplotlib
plt.rcParams["font.family"] = "Times New Roman"

class Vehicle:
    def __init__(self, current_x, current_v, current_s,vehicle_length, delta_t, lane_length, global_clock):
        self.current_x = current_x
        self.current_v = current_v
        self.current_a = 0
        self.current_s = 10000  # Initially set to a large value
        self.delta_t = delta_t
        self.vehicle_length = vehicle_length
        self.lane_length = lane_length

        self.history_t = [global_clock]
        self.history_x = [current_x]
        self.history_v = [current_v]
        self.history_a = [0]
        self.history_s = [current_s]
        self.history_delta_v = []

        # Simulate lane changing behavior, random generate when initialization (m)
        self.desired_lane_change_distance = None
        # IDM parameters
        self.v0 = 25 + np.random.normal(0,1) # Desired velocity
        self.T = 1.5 + np.random.normal(0.03)   # Safe time headway
        if self.T < 1.2:
            self.T = 1.2
        self.a = 0.3 + np.random.normal(0.01)   # Maximum acceleration
        if self.a < 0.2:
            self.a = 0.2
        self.b = 3  + np.random.normal(0.03)  # Comfortable deceleration
        if self.b < 2:
            self.b = 2
        self.delta = 4  # Acceleration exponent
        self.s0 = 3 + np.random.rand() # Minimum gap
        self.red_visual_distance = np.random.normal(100, 5)  # Distance at which vehicle will stop for red light
        self.stopped_at_red = False
        self.stop_for_red = True

    def update(self, lead_vehicle, is_red_light, global_clock):

        # Update s (distance to leading vehicle or traffic light)
        if lead_vehicle:
            self.current_s = lead_vehicle.current_x - self.current_x - lead_vehicle.vehicle_length
        elif is_red_light and self.stop_for_red:
            self.current_s = max(0, self.lane_length - self.current_x)
        else:
            self.current_s = 10000  # Large value for free driving

        # Calculate acceleration using IDM
        lead_v = lead_vehicle.current_v if lead_vehicle else self.v0
        if is_red_light and self.stop_for_red and self.current_s < self.red_visual_distance :  # Close to red light
            lead_v = 0
        delta_v = self.current_v - lead_v
        s_star = (self.s0 + max(0, self.current_v * self.T + 
                  (self.current_v * delta_v) / 
                  (2 * np.sqrt(self.a * self.b))))
        
        self.current_a = self.a * (1 - (self.current_v / self.v0)**self.delta - (s_star / max(self.current_s, 0.01))**2)
        if self.current_a < -8:
            self.current_a = -8
        if self.current_a > 3:
            self.current_a = 3
        # Update velocity and position
        self.current_v = max(0, self.current_v + self.current_a * self.delta_t)
        self.current_x = min(self.lane_length, self.current_x + self.current_v * self.delta_t)

        # Save to history
        self.history_t.append(global_clock)
        self.history_x.append(self.current_x)
        self.history_v.append(self.current_v)
        self.history_a.append(self.current_a)
        self.history_s.append(self.current_s)
        self.history_delta_v.append(delta_v)


def run_simulation(lane_length, red_light_time, green_light_time, simulation_time, delta_t, mean_headway,mean_speed, lane_change_prob, mean_vehicle_length = 5):
    # Initialize variables
    mean_vehicle_length = 5
    # mean_speed = 10 # m/s

    vehicles = []
    exited_vehicles = []
    global_clock = 0
    next_vehicle_time = np.random.normal(mean_headway, 2)
    queue_record = [] # Queue of vehicles whose speed is 0
    num_vehicles = [] # Number of vehicles in the lane
    
    # Simulation loop
    while global_clock < simulation_time:
        light_cycle = red_light_time + green_light_time
        is_red_light = (global_clock % light_cycle) < red_light_time

        # Generate new vehicle if it's time
        if global_clock >= next_vehicle_time:
            vehicle_length = np.random.normal(mean_vehicle_length, 0.5)
            
            if vehicles:
                # Consider the tail car's speed and position
                tail_car = vehicles[-1]
                current_v = np.random.normal(tail_car.current_v, 1)
                # Position the new car behind the tail car, allowing negative positions
                if tail_car.current_x - tail_car.vehicle_length < 0:
                    current_x = tail_car.current_x - tail_car.vehicle_length - tail_car.s0 - 0.2 * vehicle_length
                    current_s = tail_car.s0 + 0.2 * vehicle_length
                else:
                    current_x = 0
                    current_s = tail_car.current_x - tail_car.vehicle_length
            else:
                current_v = np.random.normal(mean_speed, 5)
                if current_v < 0:
                    current_v = 1
                current_x = 0
                current_s = 10000
            new_vehicle = Vehicle(current_x, current_v, current_s,vehicle_length, delta_t, lane_length, global_clock)
            if np.random.rand() < lane_change_prob:
                # select a distance that the in the last half of the lane distance
                new_vehicle.desired_lane_change_distance = lane_length - np.random.randint(0, int(lane_length/2))
            vehicles.append(new_vehicle)
            next_vehicle_time = global_clock + np.random.normal(mean_headway, 1)
        global_clock += delta_t
        # Update vehicles and check for exits
        i = 0
        queue_num = 0
        while i < len(vehicles):
            vehicle = vehicles[i]
            lead_vehicle = vehicles[i-1] if i > 0 else None
            vehicle.update(lead_vehicle, is_red_light, global_clock)
            # Check if vehicle has exited the lane
            if vehicle.current_x >= lane_length:
                exited_vehicles.append(vehicles.pop(i))
            elif vehicle.desired_lane_change_distance is not None and vehicle.current_x > vehicle.desired_lane_change_distance :
                exited_vehicles.append(vehicles.pop(i))
            else:
                i += 1
            # Check if vehicle is stopped
            if vehicle.current_v == 0:
                queue_num += 1
        queue_record.append(queue_num)
        num_vehicles.append(len(vehicles))
    queue_record = np.array(queue_record)
    num_vehicles = np.array(num_vehicles) 

    return vehicles, exited_vehicles, queue_record, num_vehicles

def create_time_space_diagram(vehicles, lane_length, simulation_time, delta_t,delta_x):
    # Set up the grid
    time_resolution = delta_t # seconds per pixel
    space_resolution = delta_x  # meters per pixel
    time_pixels = int(simulation_time / time_resolution)
    space_pixels = int(lane_length / space_resolution)
    
    # Create an empty grid
    time_space_mask = np.zeros((space_pixels, time_pixels), dtype=int)
    time_space_mask_occ_effect = np.zeros((space_pixels, time_pixels), dtype=int) # this is for simulate the effect of the vehicle length on the space (some part of the vehicle cannot be observed)
    vehicle_trajs = []
    for v_id,vehicle in enumerate(vehicles):
        vehicle_length_pixels = int(vehicle.vehicle_length / space_resolution)
        for t, x, v in zip(vehicle.history_t, vehicle.history_x, vehicle.history_v):
            time_idx = int(t / time_resolution) - 1
            space_idx = int(x / space_resolution) - 1
            if 0 <= space_idx < space_pixels:
                vehicle_trajs.append((v_id, time_idx, space_idx, v))
                # Mark the space occupied by the vehicle
                length_decay = x / (lane_length - 20)
       
                preserved_pixels_occ = int(vehicle_length_pixels * length_decay) - np.random.randint(0,6)
                if preserved_pixels_occ < 0:
                    preserved_pixels_occ = 0
                for i in range(vehicle_length_pixels):
                    if 0 <= space_idx - i < space_pixels:
                        time_space_mask[space_idx - i, time_idx] = 1

                for i in range(preserved_pixels_occ):
                    if 0 <= space_idx - i < space_pixels:
                        time_space_mask_occ_effect[space_idx - i, time_idx] = 1
                        
    return time_space_mask,time_space_mask_occ_effect,vehicle_trajs

# Randomly add some regular patches with angles to block the time_space_mask to simulate the occlusion events
def add_occlusions(binary_mask):
    # Convert binary mask to uint8 type (required for OpenCV operations)
    occluded_mask = (binary_mask * 1).astype(np.uint8)
    occlusion_mask = np.zeros_like(occluded_mask) # indicating the occlusion area where the binary mask is 1 and the occlusion is 0
    num_occlusions = occluded_mask.shape[1] // 20
    height, width = occluded_mask.shape
    for _ in range(num_occlusions):
        # Random center point
        center = (np.random.randint(0, width), np.random.randint(0, height))
        
        # Random width and height (between 5% and 20% of image dimensions)
        rect_width = int(np.random.normal(12,2))
        rect_height = int(np.random.normal(60,3))
        
        # Random angle (0 to 180 degrees)
        angle = np.random.randint(0, 45)
        
        # Create rotated rectangle
        rect = ((center[0], center[1]), (rect_width, rect_height), angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        # Fill the rotated rectangle with white (255)
        cv2.fillPoly(occlusion_mask, [box], 1)
        cv2.fillPoly(occluded_mask, [box], 0)
    # find common area of occlusion and binary mask
    occlusion_mask = occlusion_mask * binary_mask
    
    return occluded_mask,occlusion_mask

# add noise to the occulded diagram
def add_noise(binary_mask,emit_rate = 0.1,noise_rate = 0.005):
    binary_mask = binary_mask.copy()
    emit_mask = np.random.choice([0, 1], size=binary_mask.shape, p=[1-emit_rate, emit_rate])
    binary_mask[emit_mask == 1] = 0
    noise_mask = np.random.choice([0, 1], size=binary_mask.shape, p=[1-noise_rate, noise_rate])
    binary_mask[noise_mask == 1] = 1

    return binary_mask
    