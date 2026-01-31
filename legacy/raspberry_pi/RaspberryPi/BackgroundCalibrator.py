import pygame
import numpy as np

class BackgroundCalibrator:
    def __init__(self, screen, background_surface):
        self.screen = screen
        self.background_surface = background_surface
        self.calibration_mode = False
        self.space_pressed = False
        
        # Foreground (point cloud) parameters
        self.zoom_fore = 1.0
        self.offset_fore = np.array([0, 0]).astype(float)
        self.rotation_fore = 0
        
        # Background (image) parameters
        self.zoom_bck = 1.0
        self.offset_bck = np.array([0, 0]).astype(float)
        self.rotation_bck = 0
        
        self.thred_map = None
        self.background_image = None
        
    def load_data(self):
        # Load thred_map.npy and convert to point cloud
        self.thred_map = np.load('./config_files/thred_map.npy')
        self.point_cloud = self.thred_map_to_point_cloud(self.thred_map)
        
        # Load background image
        self.background_image = pygame.image.load('./config_files/background.png').convert_alpha()
    
    def thred_map_to_point_cloud(self, thred_map):
        # Convert thred_map to point cloud
        # This is a placeholder - implement the actual conversion logic
        return np.array([[x, y] for x in range(thred_map.shape[0]) for y in range(thred_map.shape[1]) if thred_map[x, y] > 0])
    
    def toggle_calibration_mode(self):
        self.calibration_mode = not self.calibration_mode
        if self.calibration_mode:
            self.load_data()
    
    def handle_events(self, event):
        if not self.calibration_mode:
            return
        
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            self.space_pressed = True
        elif event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
            self.space_pressed = False
        
        if event.type == pygame.MOUSEMOTION:
            if event.buttons[0]:  # Left mouse button
                dx, dy = event.rel
                if self.space_pressed:
                    self.offset_bck += np.array([dx, dy])
                else:
                    self.offset_fore += np.array([dx, dy])
            elif event.buttons[1]:  # Middle mouse button
                dx, _ = event.rel
                if self.space_pressed:
                    self.rotation_bck += dx * 0.01
                else:
                    self.rotation_fore += dx * 0.01
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Scroll up
                if self.space_pressed:
                    self.zoom_bck *= 1.1
                else:
                    self.zoom_fore *= 1.1
            elif event.button == 5:  # Scroll down
                if self.space_pressed:
                    self.zoom_bck /= 1.1
                else:
                    self.zoom_fore /= 1.1
    
    def draw(self):
        if not self.calibration_mode:
            return
        
        # Draw background image
        rotated_bg = pygame.transform.rotate(self.background_image, -np.degrees(self.rotation_bck))
        scaled_bg = pygame.transform.scale(rotated_bg, 
                                           (int(rotated_bg.get_width() * self.zoom_bck),
                                            int(rotated_bg.get_height() * self.zoom_bck)))
        bg_rect = scaled_bg.get_rect(center=(self.screen.get_width()/2 + self.offset_bck[0],
                                             self.screen.get_height()/2 + self.offset_bck[1]))
        self.background_surface.blit(scaled_bg, bg_rect)
        
        # Draw point cloud
        for point in self.point_cloud:
            screen_point = self.world_to_screen(point, self.zoom_fore, self.offset_fore, self.rotation_fore, self.screen.get_height())
            pygame.draw.circle(self.screen, (255, 0, 0), screen_point, 2)
    
    def world_to_screen(self, point, zoom, offset, rotation, screen_height):
        # Rotate
        x, y = point
        cos_theta, sin_theta = np.cos(rotation), np.sin(rotation)
        x_rot = x * cos_theta - y * sin_theta
        y_rot = x * sin_theta + y * cos_theta
        
        # Scale and translate
        x_screen = x_rot * zoom + offset[0] + self.screen.get_width() / 2
        y_screen = screen_height - (y_rot * zoom + offset[1] + screen_height / 2)
        
        return int(x_screen), int(y_screen)
    
    def finalize_calibration(self):
        # Compute the combined transformation
        self.zoom = self.zoom_fore / self.zoom_bck
        self.offset = (self.offset_fore - self.offset_bck) / self.zoom_bck
        self.rotation = self.rotation_fore - self.rotation_bck
        
        # Reset calibration mode
        self.calibration_mode = False
        
        return self.zoom, self.offset, self.rotation
