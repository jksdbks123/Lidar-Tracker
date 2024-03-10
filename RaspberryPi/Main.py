from multiprocessing import Process, Queue, Event, Manager
import time
from multiprocessing import set_start_method
from Utils import *
from GenBckFile import *
from LiDARBase import *
from MOT_TD_BCKONLIONE import MOT

class LidarVisualizer:
    def __init__(self,point_cloud_queue, tracking_result_queue,tracking_parameter_dict,tracking_param_update_event,width=1000, height=800, title='LiDAR Data Visualization'):
        pygame.init()
        pygame.font.init() 
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        pygame.font.init()  # Initialize the font module
        self.object_label_font = pygame.font.SysFont('Comic Sans MS', 20)
        
        self.running = True
        self.point_cloud_queue = point_cloud_queue # point cloud queue
        self.tracking_result_queue = tracking_result_queue
        self.tracking_parameter_dict = tracking_parameter_dict
        self.tracking_param_update_event = tracking_param_update_event
        self.catch_background = False
        self.thred_map_loaded = False
        self.if_objid = False
        self.background_data = [] 
        self.background_data_process = None
        self.tracking_prcess = None
        self.tracking_process_stop_event = None
        self.thred_map = None

        if os.path.exists(r'./thred_map.npy'):
            self.thred_map = np.load(r'./thred_map.npy')

        self.mot = None
        self.zoom = 1.0
        self.offset = np.array([500, 300])
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        self.any_slider_active = False  # Track if any slider is active
        """
        Add badget steps:
        1. Add badget object in initialization part
        2. Add the callback function for the badget (optional)
        3. Draw the badget in the initialization part
        4. Add interactive event handling in the handle_events
        """
        # Badgets
        # left upper
        self.switch_bck_recording_mode = ToggleButton(self.screen, (20, 20, 100, 30), 'Record Frames', 'Generating Backgroud', self.toggle_catch_background)
        self.switch_tracking_mode = ToggleButton(self.screen, (20, 80, 100, 30), 'Track Off', 'Track On', self.toggle_tracking_mode)
        self.switch_object_id = ToggleButton(self.screen, (140, 80, 50, 30), 'ObjID Off', 'ObjID On', self.toggle_objid_switch)
        self.switch_foreground_mode = ToggleButton(self.screen, (20, 140, 100, 30), 'Raw Point Cloud', 'Foreground Points', self.toggle_foreground)
        
        # right upper
        self.bck_length_info = InfoBox(self.screen,(850,20,100,30),'No bck info')
        self.gen_bck_bottom = Button(self.screen,(850,80,100,30),'Gen Bck',self.start_background_generation)
        # bottom
        
        self.density_slider = Slider(self.screen, (750, 750, 200, 20), "PC density",0,1,default_value=1)
        self.frame_process_time_info = InfoBox(self.screen,(850,690,100,30),'0ms')
        self.db_window_width_slider =  Slider(self.screen, (50, 750, 200, 20), "eps_width",2,30, default_value=0.4, if_int = True, if_odd = True)
        self.db_window_height_slider =  Slider(self.screen, (300, 750, 200, 20), "eps_height",2,30, default_value=0.2, if_int = True, if_odd = True)
        self.db_min_samples_slider =  Slider(self.screen, (50, 690, 200, 20), "min_samples",2,50, default_value=0.1, if_int = True)
        self.db_eps_dis_slider =  Slider(self.screen, (300, 690, 200, 20), "eps_dis",0,5, default_value=0.2)
        self.update_tracking_param_buttom = Button(self.screen,(550,750,100,30),'Update Param',self.update_tracking_param)
        
        self.info_boxes = [self.bck_length_info,self.frame_process_time_info]
        self.slider_bars = [self.db_window_width_slider,self.db_window_height_slider,self.db_min_samples_slider,self.db_eps_dis_slider,self.density_slider]
        self.events_handle_items = [self.switch_bck_recording_mode,self.switch_tracking_mode,self.switch_foreground_mode,self.switch_object_id,self.update_tracking_param_buttom,self.gen_bck_bottom] # buttoms and toggles
        self.toggle_buttons = [self.switch_bck_recording_mode,self.switch_foreground_mode,self.switch_tracking_mode] 
        # Background parameters
        self.bck_radius = 0.9

    def handle_events(self):
        self.any_slider_active = False  # Reset the flag at the start of each event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    self.running = False
            for item in self.events_handle_items:
                item.handle_event(event)

            # Handle slider events and check if any slider is being dragged
            if  self.density_slider.handle_event(event) or self.db_window_height_slider.handle_event(event) or self.db_window_width_slider.handle_event(event) or self.db_min_samples_slider.handle_event(event) or self.db_eps_dis_slider.handle_event(event):
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

    def draw(self, data,point_label = None,tracking_dic = None):
        self.screen.fill((0, 0, 0))
        data = (data.T * self.zoom + self.offset[:, None]).T
        
        if point_label is not None:
            for coord,l in zip(data,point_label):
                x,y = coord
                color_vec = color_map[l%len(color_map)]
                pygame.draw.circle(self.screen, tuple(color_vec), (x, y), 2)

            for key in tracking_dic.keys():
                his_coords = np.array(tracking_dic[key].post_seq[-10:])
                his_coords = (his_coords[:,0].reshape(-1,2) + his_coords[:,1].reshape(-1,2))/2
                his_coords[:,0] = his_coords[:,0] * self.zoom +  self.offset[0]
                his_coords[:,1] = his_coords[:,1] * self.zoom +  self.offset[1]
                color_vec = color_map[key%len(color_map)]
                for coord in his_coords:
                    pygame.draw.circle(self.screen, tuple(color_vec), (coord[0], coord[1]), 4)
                    
        if tracking_dic is not None and self.if_objid:
            for key in tracking_dic.keys():
                cur_traj = tracking_dic[key].post_seq[-1] # -1 happens here sometimes
                label_surface = self.object_label_font.render(str(key), False, (255,65,212))
                x,y = (cur_traj[0][0][0] + cur_traj[1][0][0])/2 , (cur_traj[0][1][0] + cur_traj[1][1][0])/2
                label_pos = (x * self.zoom + self.offset[0],y * self.zoom + self.offset[1])
                self.screen.blit(label_surface,label_pos)

        if point_label is None and tracking_dic is None:
            color = int(255)  # Using the slider value for RGB intensity
            for x, y in data:
                pygame.draw.circle(self.screen, (color,color,color), (x, y), 2)

        for item in self.events_handle_items:
            item.draw()
        for item in self.info_boxes:
            item.draw()
        for item in self.slider_bars:
            item.draw() 

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
                win_size = [self.db_window_height_slider.out_value,self.db_window_width_slider.out_value]
                eps_dis = self.db_eps_dis_slider.out_value
                min_samples = self.db_min_samples_slider.out_value
                self.tracking_parameter_dict['win_size'] = win_size
                self.tracking_parameter_dict['min_samples'] = min_samples
                self.tracking_parameter_dict['eps'] = eps_dis
                self.mot = MOT(self.tracking_parameter_dict, thred_map = self.thred_map, missing_thred = 10)
                self.tracking_prcess = Process(target=track_point_clouds, args=(self.tracking_process_stop_event,self.mot,self.point_cloud_queue,self.tracking_result_queue,self.tracking_parameter_dict,self.tracking_param_update_event))
                self.tracking_prcess.start()
        else:
            if self.tracking_prcess and self.tracking_prcess.is_alive():
                self.mot = None
                self.tracking_process_stop_event.set()
                self.tracking_prcess.join()
                self.tracking_prcess = None
                while not self.tracking_result_queue.empty():
                    self.tracking_result_queue.get()
                print('Tracking Terminated...')


    def toggle_foreground(self,state):
        if state:
            self.deactivate_other_toggles(self.switch_foreground_mode)
        print('Show Foreground Points')

    def toggle_objid_switch(self,state):
        self.if_objid = ~self.if_objid

    def start_background_generation(self):
        if self.background_data:
            # Ensure there's no active background process running
            if self.background_data_process and self.background_data_process.is_alive():
                print("A background generation process is already running.")
                return
            # Start the background generation process
            self.background_data_process = Process(target=generate_and_save_background, args=(self.background_data,))
            self.background_data_process.start()
            self.thred_map = np.load(r'./thred_map.npy')
            print("Started background generation process.")
        else:
            print("No background data to process.")

    def update_tracking_param(self):
        if self.tracking_prcess:

            win_size = [self.db_window_height_slider.out_value,self.db_window_width_slider.out_value]
            eps_dis = self.db_eps_dis_slider.out_value
            min_samples = self.db_min_samples_slider.out_value

            self.tracking_parameter_dict['win_size'] = win_size
            self.tracking_parameter_dict['min_samples'] = min_samples
            self.tracking_parameter_dict['eps'] = eps_dis
            self.tracking_param_update_event.set()

    def update(self):
        # Check if the background process has finished and thred_map is not loaded yet
        if self.background_data_process and not self.background_data_process.is_alive() and not self.thred_map_loaded:
            self.thred_map = np.load(r'./thred_map.npy')
            self.thred_map_loaded = True
            print("thred_map.npy loaded")
        

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
        return point_cloud_data,None,None
    
    def get_foreground_point_cloud(self,Td_map,density):
        Foreground_map = ~(np.abs(Td_map - self.thred_map) <= self.bck_radius).any(axis = 0)
        Foreground_map = Foreground_map.astype(int)
        point_cloud_data,labels = get_pcd_colored(Td_map,Foreground_map)                    
        ds_point_cloud_data_ind = np.random.choice(np.arange(len(point_cloud_data)), size = int(len(point_cloud_data) * density),replace = False).astype(int)
        point_cloud_data = point_cloud_data[ds_point_cloud_data_ind]
        labels = labels[ds_point_cloud_data_ind]
        return point_cloud_data,labels,None
    
    def run(self):

        while self.running:
            self.handle_events()
            self.update()
            density = self.density_slider.value

            if not self.point_cloud_queue.empty():
                
                if self.switch_bck_recording_mode.state:
                    Td_map = self.point_cloud_queue.get()
                    self.background_data.append(Td_map)
                    self.bck_length_info.update_text(f"Data Length: {len(self.background_data)}")
                    point_cloud_data,point_label,tracking_dic = self.get_ordinary_point_cloud(Td_map,density)
                    
                elif self.switch_foreground_mode.state:
                    Td_map = self.point_cloud_queue.get()
                    point_cloud_data,point_labels,tracking_dic = self.get_foreground_point_cloud(Td_map,density)

                elif self.switch_tracking_mode.state:
                    while True:
                        if not self.tracking_result_queue.empty():
                            Tracking_pool,Labeling_map,Td_map,time_consumption = self.tracking_result_queue.get()
                            point_cloud_data,point_labels = get_pcd_tracking(Td_map,Labeling_map,Tracking_pool)
                            tracking_dic = Tracking_pool
                            self.frame_process_time_info.update_text(f'{time_consumption:.1f}ms')
                            break
                        
                else: # default
                    Td_map = self.point_cloud_queue.get()
                    point_cloud_data,point_labels,tracking_dic = self.get_ordinary_point_cloud(Td_map,density)
                    
                self.screen.fill((0, 0, 0))
                self.draw(point_cloud_data,point_labels,tracking_dic)  

                      
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

def main():
    # pcap_file_path = r'/Users/zhihiuchen/Documents/Data/2019-12-21-7-30-0.pcap'
    pcap_file_path = r'D:\LiDAR_Data\US395.pcap'
    try:
        with Manager() as manger:
            # set_start_method('fork',force=True)
            raw_data_queue = manger.Queue() # Packet Queue
            point_cloud_queue = manger.Queue()
            tracking_result_queue = manger.Queue() # this is for the tracking results (pt,...)
            tracking_parameter_dict = manger.dict({})
            tracking_param_update_event = Event()
            # Creating processes for Core 2 and Core 3 tasks
            
            # eth_reader = load_pcap(pcap_file_path)

            packet_reader_process = Process(target=read_packets, args=(raw_data_queue,pcap_file_path,))
            packet_parser_process = Process(target=parse_packets, args=(raw_data_queue, point_cloud_queue,))
            packet_reader_process.start()
            packet_parser_process.start()

            # Running the visualization (Core 1 task) in the main process
            
            visualizer = LidarVisualizer(point_cloud_queue,tracking_result_queue,tracking_parameter_dict,tracking_param_update_event)
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

if __name__ == '__main__':
    main()