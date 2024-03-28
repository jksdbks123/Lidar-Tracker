import pygame
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import cv2
import dpkt
from sklearn.cluster import DBSCAN
import time
import socket
import math
from shapely.geometry import LineString
from shapely.ops import unary_union


class LaneDrawer:
    def __init__(self):
        self.current_lane_points = []  # List of points defining the current lane centerline
        self.current_lane_widths = []

        self.lane_points = []  # Finished list of lanes, where each lane is a list of points
        self.lane_widths = []
        self.lane_width = 12 * 0.3048  # Default lane width in feet
        self.drawing_lanes = False # mode on
        self.start_drawing_lanes = False # start a drawing session
        self.current_lane_connection = None
        

class BarDrawer:
    def __init__(self):

        self.lines = []
        self.line_counts = []
        if os.path.exists(r'./lines.npy'):
            lines = np.load(r'./lines.npy')
            for line in lines:
                self.lines.append((tuple(line[0]),tuple(line[1])))
                self.line_counts.append(0) 

        self.current_line_start = None
        self.drawing_lines = False # mode on
        self.start_drawing_lines = False # currently in a line drawing session
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

def adjust_for_zoom_and_offset(points, zoom, offset):
    adjusted_points = []
    for point in points:
        adjusted_x = (point[0] * zoom) + offset[0]
        adjusted_y = (point[1] * zoom) + offset[1]
        adjusted_points.append((adjusted_x, adjusted_y))
    return adjusted_points

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
    return exterior

    
