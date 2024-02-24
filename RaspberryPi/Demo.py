from multiprocessing import Process, Queue
import pygame
import numpy as np
import time
from multiprocessing import set_start_method

"""
raw_data_queue: UDP packets from LiDAR snesor 
LidarVisualizer.point_cloud_queue: parsed point cloud frames 

"""


# Simulated function to continuously read packets (Simulating Core 2)
def read_packets(raw_data_queue):
    while True:
        # Simulate reading a packet from the Ethernet
        raw_packet = np.random.rand(20000,2) * 600  # Placeholder for actual packet data
        raw_data_queue.put(raw_packet)
        
        time.sleep(0.05)  # Simulate delay
# Function to parse packets into point cloud data (Simulating Core 3)
def parse_packets(raw_data_queue, point_cloud_queue):
    while True:
        if not raw_data_queue.empty():
            raw_packet = raw_data_queue.get()
            # Placeholder for parsing logic; here we just pass the data through
            parsed_data = raw_packet  # In reality, you would parse the packet
            point_cloud_queue.put(parsed_data)
            time.sleep(0.05)  # Simulate processing time
            

class Slider:
    def __init__(self, screen, position, label, default_value=0.5):
        self.screen = screen
        self.position = position  # (x, y, width, height)
        self.label = label
        self.value = default_value
        self.dragging = False
        self.font = pygame.font.Font(None, 15)
        self.slider_rect = pygame.Rect(position[0] + position[2] * self.value - 5, position[1] - 5, 10, position[3] + 10)

    def draw(self):
        # Draw the label
        label_surface = self.font.render(self.label, True, (255, 255, 255))
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
            return True  # Indicates that the slider is being dragged
        return False

class LidarVisualizer:
    def __init__(self,point_cloud_queue, width=800, height=600, title='LiDAR Data Visualization'):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.running = True
        self.point_cloud_queue = point_cloud_queue # point cloud queue

        self.zoom = 1.0
        self.offset = np.array([0, 0])
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        self.any_slider_active = False  # Track if any slider is active
        
        # Sliders for parameters
        self.color_intensity_slider = Slider(self.screen, (50, 550, 200, 20), "eps", default_value=0.5)
        self.density_slider = Slider(self.screen, (300, 550, 200, 20), "min_points", default_value=0.5)

    def handle_events(self):
        self.any_slider_active = False  # Reset the flag at the start of each event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
            
            # Handle slider events and check if any slider is being dragged
            if self.color_intensity_slider.handle_event(event) or self.density_slider.handle_event(event):
                self.any_slider_active = True
                continue  # Skip other event handling if a slider is active

            # Block panning and zooming when a slider is being adjusted
            if not self.any_slider_active:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button for dragging
                        self.dragging = True
                        self.last_mouse_pos = event.pos
                    elif event.button == 4:  # Mouse wheel up to zoom in
                        self.zoom *= 1.1
                    elif event.button == 5:  # Mouse wheel down to zoom out
                        self.zoom /= 1.1
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Stop dragging on left mouse button release
                        self.dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        mouse_pos = event.pos
                        movement = np.array(mouse_pos) - np.array(self.last_mouse_pos)
                        self.offset += movement
                        self.last_mouse_pos = mouse_pos

    def draw(self, data):
        self.screen.fill((0, 0, 0))
        data = (data.T * self.zoom + self.offset[:, None]).T
        color = int(self.color_intensity_slider.value * 255)  # Using the slider value for RGB intensity
        for x, y in data:
            pygame.draw.circle(self.screen, (color,color,color), (x, y), 2)
        self.color_intensity_slider.draw()
        self.density_slider.draw()
        pygame.display.flip()
    
    def process_lidar_data(self):
        # Instead of simulating data, pull from the parsed_data_queue
        if not self.parsed_data_queue.empty():
            data = self.parsed_data_queue.get()
            return data
        else:
            # Return some default data if the queue is empty to avoid errors
            return np.array([[0], [0]])

    def run(self):
        while self.running:
            self.handle_events()
           
            if not self.point_cloud_queue.empty():
                point_cloud_data = self.point_cloud_queue.get()
                
                density = self.density_slider.value
                ds_point_cloud_data_ind = np.random.choice(np.arange(len(point_cloud_data)), size = int(len(point_cloud_data) * density),replace = False).astype(int)
                point_cloud_data = point_cloud_data[ds_point_cloud_data_ind]
                
                # color = self.color_intensity_slider.value
                # point_cloud_data # 
                
                self.screen.fill((0, 0, 0))  # Clear screen
                self.draw(point_cloud_data)


    def quit(self):
        self.running = False
        pygame.quit()

if __name__ == '__main__':
    set_start_method('fork')
    raw_data_queue = Queue() # Packet Queue
    point_cloud_queue = Queue()
    # Creating processes for Core 2 and Core 3 tasks
    packet_reader_process = Process(target=read_packets, args=(raw_data_queue,))
    packet_parser_process = Process(target=parse_packets, args=(raw_data_queue, point_cloud_queue,))

    packet_reader_process.start()
    packet_parser_process.start()
    
    # Running the visualization (Core 1 task) in the main process
    visualizer = LidarVisualizer(point_cloud_queue)
    visualizer.run()
    
    # Cleanup
    packet_reader_process.terminate()
    packet_parser_process.terminate()

    visualizer.quit()
