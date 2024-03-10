import pygame
import numpy as np
import os
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import cv2
import dpkt
from sklearn.cluster import DBSCAN
import time

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



