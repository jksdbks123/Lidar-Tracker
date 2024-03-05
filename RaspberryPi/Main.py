from multiprocessing import Process, Queue, Event, Manager

import time
from multiprocessing import set_start_method
from Utils import *
from GenBckFile import *
from LiDARBase import *
from MOT_TD_BCKONLIONE import MOT

"""
raw_data_queue: UDP packets from LiDAR snesor 
LidarVisualizer.point_cloud_queue: parsed point cloud frames 
"""

def generate_and_save_background(background_data):
    thred_map = gen_bckmap(np.array(background_data), N = 10,d_thred = 0.1,bck_n = 3)
    np.save('./thred_map.npy',thred_map)
    print('Generate Bck')

def track_point_clouds(stop_event,mot,point_cloud_queue,result_queue):
    while not stop_event.is_set():
        if not point_cloud_queue.empty():
            Td_map =  point_cloud_queue.get()
            # some steps
            if not mot.if_initialized:
                mot.initialization(Td_map)
                Tracking_pool = mot.Tracking_pool
                Labeling_map = mot.cur_Labeling_map
            else:
                mot.mot_tracking_step(Td_map)
                Tracking_pool = mot.Tracking_pool
                Labeling_map = mot.cur_Labeling_map

            result_queue.put((Tracking_pool,Labeling_map,Td_map))
            # print('tracking now...',result_queue.empty(), point_cloud_queue.empty())
            # time.sleep(0.5)


    print('Terminated tracking process')

class LidarVisualizer:
    def __init__(self,point_cloud_queue, tracking_result_queue,width=800, height=600, title='LiDAR Data Visualization'):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)

        self.running = True
        self.point_cloud_queue = point_cloud_queue # point cloud queue
        # self.point_cloud_queue_4track = point_cloud_queue_4track
        self.tracking_result_queue = tracking_result_queue
        self.catch_background = False
        self.background_data = [] 
        self.background_data_process = None
        self.tracking_prcess = None
        self.tracking_process_stop_event = None
        self.thred_map = None
        if os.path.exists(r'./thred_map.npy'):
            self.thred_map = np.load(r'./thred_map.npy')
        self.mot = MOT(win_size = [7,13], eps = 1.5, min_samples = 5, thred_map = self.thred_map, missing_thred = 10)

        

        self.zoom = 1.0
        self.offset = np.array([0, 0])
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        self.any_slider_active = False  # Track if any slider is active
        
        # Badgets
        self.color_intensity_slider = Slider(self.screen, (50, 550, 200, 20), "eps", default_value=0.5)
        self.density_slider = Slider(self.screen, (300, 550, 200, 20), "min_points", default_value=0.5)
        self.switch_bck_recording_mode = ToggleButton(self.screen, (20, 20, 100, 50), 'Record Frames', 'Generating Backgroud', self.toggle_catch_background)
        self.switch_tracking_mode = ToggleButton(self.screen, (20, 100, 100, 50), 'Track Off', 'Track On', self.toggle_tracking_mode)
        self.switch_foreground_mode = ToggleButton(self.screen, (20, 180, 100, 50), 'Raw Point Cloud', 'Foreground Points', self.toggle_foreground)
        self.bck_length_info = InfoBox(self.screen,(650,20,100,50),'No bck info')
        self.gen_bck_bottom = Button(self.screen,(650,100,100,50),'Gen Bck',self.start_background_generation)
        self.toggle_buttons = [self.switch_bck_recording_mode,self.switch_foreground_mode,self.switch_tracking_mode] 
        self.bck_radius = 0.9

    def handle_events(self):
        self.any_slider_active = False  # Reset the flag at the start of each event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False

            self.switch_bck_recording_mode.handle_event(event)
            self.switch_tracking_mode.handle_event(event)
            self.switch_foreground_mode.handle_event(event)
            self.gen_bck_bottom.handle_event(event)
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

    def draw(self, data,color_label = None):
        self.screen.fill((0, 0, 0))
        data = (data.T * self.zoom + self.offset[:, None]).T
        if color_label is not None:
            for coord,l in zip(data,color_label):
                x,y = coord
                color_vec = color_map[l%len(color_map)]
                pygame.draw.circle(self.screen, tuple(color_vec), (x, y), 2)
        else:
            color = int(self.color_intensity_slider.value * 255)  # Using the slider value for RGB intensity
            for x, y in data:
                pygame.draw.circle(self.screen, (color,color,color), (x, y), 2)

        self.color_intensity_slider.draw()
        self.density_slider.draw()
        self.switch_bck_recording_mode.draw() 
        self.switch_tracking_mode.draw()
        self.switch_foreground_mode.draw()
        
        self.bck_length_info.draw()
        self.gen_bck_bottom.draw()
        pygame.display.flip()
    
    def deactivate_other_toggles(self,activate_button):
        for button in self.toggle_buttons:
            if button != activate_button:
                button.set_state()

    def toggle_catch_background(self,state):
        if state:
            self.deactivate_other_toggles(self.switch_bck_recording_mode)
        print('Test_toggle_bck')

    def toggle_tracking_mode(self,state):
        if state:
            self.deactivate_other_toggles(self.switch_tracking_mode)
            if self.tracking_prcess is None or not self.tracking_prcess.is_alive():
                self.tracking_process_stop_event = Event()
                self.tracking_prcess = Process(target=track_point_clouds, args=(self.tracking_process_stop_event,self.mot,self.point_cloud_queue,self.tracking_result_queue))
                self.tracking_prcess.start()
        else:
            if self.tracking_prcess and self.tracking_prcess.is_alive():
                self.tracking_process_stop_event.set()
                self.tracking_prcess.join()
                self.tracking_prcess = None
                print('Tracking Terminated...')
        print('Test_track')

    def toggle_foreground(self,state):
        if state:
            self.deactivate_other_toggles(self.switch_foreground_mode)
        print('Show Foreground Points')

    def start_background_generation(self):
        if self.background_data:
            # Ensure there's no active background process running
            if self.background_data_process and self.background_data_process.is_alive():
                print("A background generation process is already running.")
                return
            # Start the background generation process
            self.background_data_process = Process(target=generate_and_save_background, args=(self.background_data,))
            self.background_data_process.start()
            print("Started background generation process.")
        else:
            print("No background data to process.")

    def process_lidar_data(self):
        # Instead of simulating data, pull from the parsed_data_queue
        if not self.parsed_data_queue.empty():
            data = self.parsed_data_queue.get()
            return data
        else:
            # Return some default data if the queue is empty to avoid errors
            return np.array([[0], [0]])
        
    def get_ordinary_point_cloud(self,Td_map,density):
        point_cloud_data = get_pcd_uncolored(Td_map)
        ds_point_cloud_data_ind = np.random.choice(np.arange(len(point_cloud_data)), size = int(len(point_cloud_data) * density),replace = False).astype(int)
        point_cloud_data = point_cloud_data[ds_point_cloud_data_ind]
        return point_cloud_data,None
    
    def get_foreground_point_cloud(self,Td_map,density):
        Foreground_map = ~(np.abs(Td_map - self.thred_map) <= self.bck_radius).any(axis = 0)
        Foreground_map = Foreground_map.astype(int)
        point_cloud_data,labels = get_pcd_colored(Td_map,Foreground_map)                    
        ds_point_cloud_data_ind = np.random.choice(np.arange(len(point_cloud_data)), size = int(len(point_cloud_data) * density),replace = False).astype(int)
        point_cloud_data = point_cloud_data[ds_point_cloud_data_ind]
        labels = labels[ds_point_cloud_data_ind]
        return point_cloud_data,labels
    
    def run(self):

        while self.running:
            self.handle_events()
            density = self.density_slider.value

            if not self.point_cloud_queue.empty():
                
                if self.switch_bck_recording_mode.state:
                    Td_map = self.point_cloud_queue.get()
                    self.background_data.append(Td_map)
                    self.bck_length_info.update_text(f"Data Length: {len(self.background_data)}")
                    point_cloud_data,labels = self.get_ordinary_point_cloud(Td_map,density)
                    
                elif self.switch_foreground_mode.state:
                    Td_map = self.point_cloud_queue.get()
                    point_cloud_data,labels = self.get_foreground_point_cloud(Td_map,density)

                elif self.switch_tracking_mode.state:
                    # self.point_cloud_queue_4track.put(Td_map)
                    # labels = something
                    # Td_map = self.point_cloud_queue.get()
                    while True:
                        if not self.tracking_result_queue.empty():
                            Tracking_pool,Labeling_map,Td_map = self.tracking_result_queue.get()
                            point_cloud_data,labels = get_pcd_tracking(Td_map,Labeling_map,Tracking_pool)
                            break
                            
                else: # default
                    Td_map = self.point_cloud_queue.get()
                    point_cloud_data,labels = self.get_ordinary_point_cloud(Td_map,density)
                    
                self.screen.fill((0, 0, 0))
                self.draw(point_cloud_data,labels)  

                      

    def quit(self):
        self.running = False
        if self.background_data_process:
            self.background_data_process.join()  # Wait for the process to complete
        if self.tracking_prcess and self.tracking_prcess.is_alive():
            self.tracking_process_stop_event.set()
            self.tracking_prcess.join()
            self.tracking_prcess = None
            print('Tracking Terminated...')

        pygame.quit()


if __name__ == '__main__':
    pcap_file_path = r'../../../Data/2019-12-21-7-30-0.pcap'
    try:
        with Manager() as manger:
            set_start_method('fork',force=True)
            raw_data_queue = manger.Queue() # Packet Queue
            point_cloud_queue = manger.Queue()
            # point_cloud_queue_4track = manger.Queue()
            tracking_result_queue = manger.Queue() # this is for the tracking results (pt,...)
            # Creating processes for Core 2 and Core 3 tasks
            
            eth_reader = load_pcap(pcap_file_path)

            packet_reader_process = Process(target=read_packets, args=(raw_data_queue,eth_reader,))
            packet_parser_process = Process(target=parse_packets, args=(raw_data_queue, point_cloud_queue,))
            packet_reader_process.start()
            packet_parser_process.start()


            # Running the visualization (Core 1 task) in the main process
            
            visualizer = LidarVisualizer(point_cloud_queue,tracking_result_queue)
            visualizer.run()
            
            # Cleanup
            packet_reader_process.terminate()
            packet_parser_process.terminate()
            
            packet_reader_process.join()
            packet_parser_process.join()

            visualizer.quit()

    except KeyboardInterrupt :
        packet_reader_process.terminate()
        packet_parser_process.terminate()
        visualizer.tracking_process_stop_event.set()
        visualizer.tracking_prcess.terminate()
        packet_reader_process.join()
        packet_parser_process.join()
        if visualizer.tracking_prcess and visualizer.tracking_prcess.is_alive():
            visualizer.tracking_process_stop_event.set()
            visualizer.tracking_prcess.join()
            visualizer.tracking_prcess = None
            print('Tracking Terminated...')
        visualizer.tracking_prcess.join()
        visualizer.quit()

