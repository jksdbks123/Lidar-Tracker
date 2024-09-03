import pygame
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
# import cv2
import dpkt
from sklearn.cluster import DBSCAN
import time
import socket
import math
import pickle
from shapely.geometry import LineString,Point
from shapely.ops import unary_union
import geopandas as gpd
# from LiDARBase import theta
from queue import Queue, Full
from collections import deque
import threading
import multiprocessing

class LaneDrawer:
    def __init__(self):
        self.current_lane_points = []  # List of points defining the current lane centerline
        self.current_lane_widths = []
        self.lane_section_foreground_point_counts = []
        self.lane_points = []  # n x k x poly points
        self.lane_widths = [] # n x k - 1 
        self.lane_subsections_poly = [] # n x k x poly lane segments
        self.lane_end_points = []
        self.lane_width = 12 * 0.3048  # Default lane width in feet
        self.drawing_lanes = False # mode on
        self.start_drawing_lanes = False # start a drawing session
        self.current_lane_connection = None
        self.lane_gdf = None # for counting foreground points in polys

        self.lane_dic_path = r'./config_files/lane_dic.pkl'
        self.lane_width_dic_path = r'./config_files/lane_width_dic.pkl'
        self.lane_poly_path = r'./config_files/lane_poly.pkl'
        self.lane_end_points_path = r'./config_files/lane_end_points.pkl'
        # self.lnae_width_dic_path = r'./config_files/' 
        self.read_lanes()

    def remove_last_record(self):
        
        if self.lane_section_foreground_point_counts:
            self.lane_section_foreground_point_counts.pop()
        if self.lane_points:
            self.lane_points.pop()
        if self.lane_widths:
            self.lane_widths.pop()
        if self.lane_subsections_poly:
            self.lane_subsections_poly.pop()
        if self.lane_end_points:
            self.lane_end_points.pop()
        self.save()
        
    def read_lanes(self):
        
        if os.path.exists(self.lane_dic_path) and os.path.exists(self.lane_width_dic_path) and os.path.exists(self.lane_poly_path) and os.path.exists(self.lane_end_points_path):
            with open(self.lane_dic_path, "rb") as f:
                lane_dic = pickle.load(f)
                for key in lane_dic.keys():
                    self.lane_points.append(lane_dic[key])
            with open(self.lane_width_dic_path, "rb") as f:
                lane_widths = pickle.load(f)
                for key in lane_widths.keys():
                    self.lane_widths.append(lane_widths[key])
            with open(self.lane_poly_path, 'rb') as f:
                self.lane_subsections_poly = pickle.load(f)
            with open(self.lane_end_points_path, 'rb') as f:
                self.lane_end_points = pickle.load(f)

    def save(self):

        lane_dic = {}
        lane_width_dic = {}
        for i,lane_points in enumerate(self.lane_points):
            lane_dic[i] = lane_points
        for i,width_series in enumerate(self.lane_widths):
            lane_width_dic[i] = width_series
        with open(self.lane_dic_path, "wb") as f:
            pickle.dump(lane_dic, f)
        with open(self.lane_width_dic_path, "wb") as f:
            pickle.dump(lane_width_dic, f)
        with open(self.lane_poly_path, 'wb') as f:
            pickle.dump(self.lane_subsections_poly, f, pickle.HIGHEST_PROTOCOL)
        with open(self.lane_end_points_path, 'wb') as f:
            pickle.dump(self.lane_end_points, f, pickle.HIGHEST_PROTOCOL)
            
    def update_lane_gdf(self):
        if self.lane_subsections_poly:
            self.lane_gdf = get_lane_gdf(self.lane_subsections_poly)
            self.lane_unit_range_ranging_Tdmap = get_lane_unit_range_ranging_Tdmap(self.lane_gdf)
            print('Lane zone updated')

    def clear(self):
        self.current_lane_points.clear()
        self.current_lane_widths.clear()
        self.lane_points.clear()
        self.lane_widths.clear()
        self.current_lane_connection = None
        self.drawing_lanes = False
        self.lane_end_points.clear()
        self.lane_subsections_poly.clear()
        self.lane_section_foreground_point_counts.clear()
            
class BarDrawer:
    def __init__(self):

        self.lines = [] # n x [(x1,y1),(x2,y2)]
        self.line_counts = []
        self.current_line_start = None
        self.drawing_lines = False # mode on
        self.start_drawing_lines = False # currently in a line drawing session
        self.current_line_connection = None
        self.bar_path = r'./config_files/bars.txt'
        self.read_bars()
        

    def read_bars(self):
        if os.path.exists(self.bar_path):
            with open(self.bar_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    x1,y1,x2,y2 = line.split(' ')
                    self.lines.append([(float(x1),float(y1)),(float(x2),float(y2))])
                    self.line_counts.append(0)
        
    def save(self):
        with open(self.bar_path,'w') as f:
            for line in self.lines:
                x1,y1 = line[0]
                x2,y2 = line[1]
                f.writelines(f'{x1} {y1} {x2} {y2}\n')

    def clear(self):
        self.lines.clear()
        self.line_counts.clear()
        self.current_line_start = None
        self.start_drawing_lines = False
        self.current_line_connection = None
            
class Slider:
    def __init__(self, screen, position, label, min_value, max_value,default_value=0.5,if_int = False, if_odd = False):
        self.screen = screen
        self.position = position  # (x, y, width, height)
        self.if_int = if_int
        self.if_odd = if_odd
        self.min_value = min_value
        self.max_value = max_value
        self.label = label
        self.value = default_value
        self.out_value = self.min_value + (self.max_value - self.min_value) * self.value
        if self.if_int:
            self.out_value = int(self.out_value)
        if self.if_odd:
            if self.out_value % 2 == 0:
                self.out_value += 1
        self.dragging = False
        self.font = pygame.font.Font(None, 15)
        self.slider_rect = pygame.Rect(position[0] + position[2] * self.value - 5, position[1] - 5, 10, position[3] + 10)

    def draw(self):
        # Draw the label
        label_surface = self.font.render(f'{self.label}:{self.out_value:.1f}', True, (255, 255, 255))
        self.screen.blit(label_surface, (self.position[0], self.position[1] - 30))
        # Draw the bar
        pygame.draw.rect(self.screen, (100, 100, 100), self.position)
        # Draw the slider
        self.slider_rect.x = self.position[0] + self.position[2] * self.value - 5
        pygame.draw.rect(self.screen, (180, 180, 180), self.slider_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.slider_rect.collidepoint(event.pos):
            self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            x, _, width, _ = self.position
            self.value = (event.pos[0] - x) / width
            self.value = min(max(self.value, 0), 1)
            self.out_value = self.min_value + (self.max_value - self.min_value) * self.value
            if self.if_int:
                self.out_value = int(self.out_value)
            if self.if_odd:
                if self.out_value % 2 == 0:
                    self.out_value += 1
            return True  # Indicates that the slider is being dragged
        return False

class ToggleButton:
    def __init__(self, screen, position, text_off, text_on, callback):
        self.rect = pygame.Rect(position[0], position[1], position[2], position[3])
        self.screen = screen
        self.text_off = text_off
        self.text_on = text_on
        self.font = pygame.font.Font(None, 15)
        self.callback = callback
        self.state = False  # False = off, True = on
        self.color_off = (200, 200, 200)
        self.color_on = (100, 200, 100)

    def set_state(self):
        self.state = False
        self.callback(self.state)  # Call the callback function when toggled

    def draw(self):
        # Draw the button with different colors/text based on its state
        color = self.color_on if self.state else self.color_off
        text = self.text_on if self.state else self.text_off
        pygame.draw.rect(self.screen, color, self.rect)
        text_surf = self.font.render(text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(center=self.rect.center)
        self.screen.blit(text_surf, text_rect)
    
    def handle_event(self, event):
        # Toggle state if the button is clicked
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.state = not self.state
                self.callback(self.state)  # Call the callback function when toggled
    def is_mouse_over(self, pos):
        return self.rect.collidepoint(pos)

class InfoBox:
    def __init__(self, screen, position, initial_text="", background_color=(50, 50, 50), text_color=(255, 255, 255)):
        self.screen = screen
        self.rect = pygame.Rect(position[0], position[1], position[2], position[3])
        self.font = pygame.font.Font(None, 15)
        self.text = initial_text
        self.background_color = background_color
        self.text_color = text_color

    def update_text(self, new_text):
        self.text = new_text

    def draw(self):
        # Draw the background box
        pygame.draw.rect(self.screen, self.background_color, self.rect)
        # Prepare and draw the text 
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        self.screen.blit(text_surf, text_rect)

class Button:
    def __init__(self, screen, position,text, callback):
        self.screen = screen 
        self.rect = pygame.Rect(position[0],position[1],position[2],position[3])
        self.text = text
        self.font = pygame.font.Font(None, 15)
        self.callback = callback
        self.color = (200, 200, 200)
        self.text_color = (0, 0, 0)

    def draw(self):
        # Draw the button
        pygame.draw.rect(self.screen, self.color, self.rect)
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        self.screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.callback()  # Execute the callback function
                
    def is_mouse_over(self, pos):
        return self.rect.collidepoint(pos)

def line_segments_intersect(seg1_start, seg1_end, seg2_start, seg2_end):
    """Returns True if line segments seg1 and seg2 intersect."""
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    return ccw(seg1_start, seg2_start, seg2_end) != ccw(seg1_end, seg2_start, seg2_end) and ccw(seg1_start, seg1_end, seg2_start) != ccw(seg1_start, seg1_end, seg2_end)


def create_bufferzone_vertex(centerline,width): # n, n -1
    segment_buffers = []
    for i in range(len(centerline) - 1):
        segment = LineString([centerline[i], centerline[i + 1]])
        # Use width for the segment; ensure list has enough entries
        segment_width = width[min(i, len(width) - 1)] / 2
        buffer = segment.buffer(segment_width, resolution=16, cap_style=2, join_style=2)
        segment_buffers.append(buffer)
    # Merge all segment buffers into a single polygon
    merged_buffer = unary_union(segment_buffers)
    exterior = list(merged_buffer.exterior.coords)
    return exterior # list([])

def calculate_segment_lengths_and_cumulative_lengths(centerline):
    lengths = []
    cumulative_lengths = [0]  # Start with 0 length at the first point
    for i in range(len(centerline) - 1):
        segment = LineString([centerline[i], centerline[i+1]])
        length = segment.length
        lengths.append(length)
        cumulative_lengths.append(cumulative_lengths[-1] + length)
    return lengths, cumulative_lengths
def interpolate_width(cumulative_lengths, widths, position):
    # Find the segment where the position belongs
    for i in range(1, len(cumulative_lengths)):
        if position <= cumulative_lengths[i]:
            segment_start_length = cumulative_lengths[i-1]
            segment_end_length = cumulative_lengths[i]
            segment_relative_position = (position - segment_start_length) / (segment_end_length - segment_start_length)
            
            # Linearly interpolate the width based on the relative position within the segment
            width_at_position = widths[i-1] + (widths[min(i, len(widths)-1)] - widths[i-1]) * segment_relative_position
            return width_at_position
    return widths[-1]  # Return the last width if position is beyond the last segment

def create_subsection_polygons(centerline, widths, subsection_length):
    lane_line = LineString(centerline)
    total_length = lane_line.length
    subsection_polygons = []
    lengths, cumulative_lengths = calculate_segment_lengths_and_cumulative_lengths(centerline)
    
    distances = np.arange(0, total_length, subsection_length)
    for start_dist in distances:
        end_dist = min(start_dist + subsection_length, total_length)
        
        # Interpolate widths for start and end points of the subsection
        start_width = interpolate_width(cumulative_lengths, widths, start_dist)
        end_width = interpolate_width(cumulative_lengths, widths, end_dist)
        avg_width = (start_width + end_width) / 2  # Use average width for the subsection
        
        start_point = lane_line.interpolate(start_dist)
        end_point = lane_line.interpolate(end_dist)
        subsection_line = LineString([start_point, end_point])
        buffer_polygon = subsection_line.buffer(avg_width / 2, resolution=16, cap_style=2, join_style=2)
        subsection_polygons.append(buffer_polygon)

    return subsection_polygons

def get_lane_gdf(lane_subsections_poly):
    lane_polygons_with_id = []
    for lane_index, lane in enumerate(lane_subsections_poly):
        for subsection_index, poly in enumerate(lane):
            lane_polygons_with_id.append({'lane_id': lane_index, 'subsection_id': subsection_index, 'geometry': poly})
    
    # Create a GeoDataFrame for lane polygons with identifiers
    lane_gdf = gpd.GeoDataFrame(lane_polygons_with_id)
    return lane_gdf

def get_lane_unit_range_ranging_Tdmap(lane_gdf):
    # create lane_unit_range_ranging_Tdmap
    lane_unit_range_ranging_Tdmap = []

    for unit_ind in range(len(lane_gdf)):
        lane_unit = lane_gdf.iloc[unit_ind]
        x_coords,y_coords = lane_unit.geometry.exterior.coords.xy
        coords = np.c_[x_coords,y_coords]
        
        azimuth_unit = np.arctan2(coords[:,0],coords[:,1]) * 180 / np.pi
        azimuth_unit[azimuth_unit < 0] += 360
        min_azimuth = np.min(azimuth_unit)
        max_azimuth = np.max(azimuth_unit)
        min_ind = int(min_azimuth / 0.2)
        max_ind = int(max_azimuth / 0.2)

        dis = np.sum(coords**2,axis = 1)**0.5
        min_dis = np.min(dis)
        max_dis = np.max(dis)
        lane_unit_range_ranging_Tdmap.append([min_ind,max_ind,min_dis,max_dis])

    return lane_unit_range_ranging_Tdmap

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
omega = Data_order[:,0]
theta = np.sort(omega)

def get_occupation_ind(Td_map,lane_unit_range_ranging_Tdmap,count_thred,bck_radius,thred_map):
    Foreground_map = ~(np.abs(Td_map - thred_map) <= bck_radius).any(axis = 0)
    Foreground_map = Foreground_map.astype(int)
    Td_map_cos = np.cos(theta * np.pi/180).reshape(-1,1) * Td_map
    occupation_ind = []
    for min_ind,max_ind,min_dis,max_dis in lane_unit_range_ranging_Tdmap:
        if max_ind - min_ind > Td_map.shape[1]/2:
            occupation_map = np.concatenate([Foreground_map[:,:min_ind],Foreground_map[:,max_ind:]],axis = 1)
            dis_map = np.concatenate([Td_map_cos[:,:min_ind],Td_map_cos[:,max_ind:]],axis = 1)
        else:
            occupation_map = Foreground_map[:,min_ind:max_ind]
            dis_map = Td_map_cos[:,min_ind:max_ind]
        
        dis_map_ = dis_map * occupation_map
        occupation_flag = ((dis_map_ < max_dis) * (dis_map_ > min_dis)).sum() > count_thred
        occupation_ind.append(occupation_flag)
    occupation_ind = np.array(occupation_ind)


    return occupation_ind


def get_lane_section_foreground_point_counts(lane_subsections_poly,lane_gdf,point_data,point_label):
    foreground_points = point_data[point_label == 1]
    # Convert foreground points to GeoDataFrame
    points_gdf = gpd.GeoDataFrame([{'geometry': Point(xy)} for xy in foreground_points])
    
    # Perform spatial join to find which points lie within which polygons
    joined_gdf = gpd.sjoin(points_gdf, lane_gdf, how="inner", predicate="within")
    
    # Count points in each polygon, reset index to make 'index_right' a column
    counts_per_polygon = joined_gdf.groupby('index_right').size().reset_index(name='counts')
    
    # Merge counts back into lane_gdf to associate each lane and subsection with its count
    lane_gdf = lane_gdf.merge(counts_per_polygon, left_index=True, right_on='index_right', how='left').fillna(0)
    
    # Rearrange data into the desired output structure, Lane_laser_indexes
    lane_section_foreground_point_counts = []
    for lane_index in range(len(lane_subsections_poly)):
        lane_sections = lane_gdf[lane_gdf['lane_id'] == lane_index].sort_values(by='subsection_id')
        section_counts = lane_sections['counts'].tolist()
        lane_section_foreground_point_counts.append(section_counts)

    return lane_section_foreground_point_counts


def convert_coordinates_numpy(screen_height, point):
        """Converts y-coordinate to simulate origin at bottom-left."""
        return point[:,0], screen_height  - y


def rotate_points_around_center_numpy(points, angle_degrees, center):
    """Rotates an array of points around a given center."""
    angle_radians = np.radians(angle_degrees)

    # Translate points to origin (center of rotation)
    translated_points = points - center

    # Rotation matrix
    cos_angle = np.cos(angle_radians)
    sin_angle = np.sin(angle_radians)
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    # Apply rotation
    rotated_translated_points = np.dot(translated_points, rotation_matrix)

    # Translate points back
    rotated_points = rotated_translated_points + center

    return rotated_points

def screen_to_world(coords, zoom, offset, rotation_angle, screen_height):
    # Flip y-axis
    coords = coords.astype(float)
    coords[:, 1] = screen_height - coords[:, 1]
    
    # Reverse offset
    coords -= offset
    
    # Reverse zoom
    coords /= zoom
    
    # Reverse rotation around the world origin (0, 0)
    angle_radians = np.radians(-rotation_angle)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    coords = np.dot(coords, rotation_matrix)
    
    return coords


def world_to_screen(coords, zoom, offset, rotation_angle, screen_height):
    # Apply rotation around the world origin (0, 0)
    coords = coords.astype(float)
    angle_radians = np.radians(rotation_angle)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])
    coords = np.dot(coords, rotation_matrix)
    
    # Apply zoom
    coords *= zoom
    
    # Apply offset
    coords += offset
    
    # Flip y-axis
    coords[:, 1] = screen_height - coords[:, 1]
    
    return coords

