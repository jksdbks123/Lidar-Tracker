from multiprocessing import Process, Queue
import time
from multiprocessing import set_start_method
from ReadPackets import *
from Utils import *
from GenBckFile import *
"""
raw_data_queue: UDP packets from LiDAR snesor 
LidarVisualizer.point_cloud_queue: parsed point cloud frames 

"""


# # Simulated function to continuously read packets (Simulating Core 2)
def read_packets(raw_data_queue,eth_reader):
    while True:
        # Simulate reading a packet from the Ethernet
        try:
            ts,buf = next(eth_reader)
            eth = dpkt.ethernet.Ethernet(buf)
        except:
            continue
        if eth.type == 2048: # for ipv4
            if (type(eth.data.data) == dpkt.udp.UDP):# for ipv4
                data = eth.data.data.data
                packet_status = eth.data.data.sport
                if packet_status == 2368:
                    if len(data) != 1206:
                        continue
            # raw_packet = np.random.rand(20000,2) * 600  # Placeholder for actual packet data
                    raw_data_queue.put((ts,data))
# Function to parse packets into point cloud data (Simulating Core 3)
def parse_packets(raw_data_queue, point_cloud_queue):
    
    culmulative_azimuth_values = []
    culmulative_laser_ids = []
    culmulative_distances = []
    # culmulative_intensities = []
    Td_map = np.zeros((32,1800))
    # Intens_map = np.zeros((32,1800))
    next_ts = 0
    if not raw_data_queue.empty():
        
        while True:
            ts,raw_packet = raw_data_queue.get()
            
            distances,intensities,azimuth_per_block,Timestamp = parse_one_packet(raw_packet)
            next_ts = ts + 0.1
            azimuth = calc_precise_azimuth(azimuth_per_block) # 32 x 12
            culmulative_azimuth_values.append(azimuth)
            culmulative_laser_ids.append(laser_id)
            culmulative_distances.append(distances)
            break
    while True:
        while True:
            if not raw_data_queue.empty():
                ts,raw_packet = raw_data_queue.get()
                # Placeholder for parsing logic; here we just pass the data through
                distances,intensities,azimuth_per_block,Timestamp = parse_one_packet(raw_packet)
                # flag = self.if_rollover(azimuth_per_block,Initial_azimuth)
                azimuth = calc_precise_azimuth(azimuth_per_block) # 32 x 12

                if ts > next_ts:
                    if len(culmulative_azimuth_values) > 0:                             
                        culmulative_azimuth_values = np.concatenate(culmulative_azimuth_values,axis = 1)
                        culmulative_azimuth_values += Data_order[:,1].reshape(-1,1)
                        culmulative_laser_ids = np.concatenate(culmulative_laser_ids,axis = 1).flatten()
                        culmulative_distances = np.concatenate(culmulative_distances,axis = 1).flatten()
                        # culmulative_intensities = np.concatenate(culmulative_intensities,axis = 1).flatten()
                        culmulative_azimuth_inds = np.around(culmulative_azimuth_values/0.2).astype('int').flatten()
                        culmulative_azimuth_inds[(culmulative_azimuth_inds<0)|(culmulative_azimuth_inds>1799)] = culmulative_azimuth_inds[(culmulative_azimuth_inds<0)|(culmulative_azimuth_inds>1799)]%1799

                        
                        Td_map[culmulative_laser_ids,culmulative_azimuth_inds] = culmulative_distances
                        # Intens_map[culmulative_laser_ids,culmulative_azimuth_inds] = culmulative_intensities

                        point_cloud_queue.put(Td_map[arg_omega,:]) #32*1800
                    else:
                        point_cloud_queue.put(Td_map) #32*1800

                    culmulative_azimuth_values = []
                    culmulative_laser_ids = []
                    culmulative_distances = []
                    # culmulative_intensities = []

                    Td_map = np.zeros((32,1800))
                    # Intens_map = np.zeros((32,1800))
                    next_ts += 0.1
                    break
                else:
                    culmulative_azimuth_values.append(azimuth)
                    culmulative_laser_ids.append(laser_id)
                    culmulative_distances.append(distances)
                    # culmulative_intensities.append(intensities)


def toggle_catch_background(state):
    print('Test_toggle_bck')
def toggle_tracking_mode(state):
    print('Test_track')
def generate_and_save_background(background_data):
    thred_map = gen_bckmap(np.array(background_data), N = 10,d_thred = 0.1,bck_n = 3)
    np.save('./thred_map.npy',thred_map)
    print('Generate Bck')


class LidarVisualizer:
    def __init__(self,point_cloud_queue, width=800, height=600, title='LiDAR Data Visualization'):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.running = True
        self.point_cloud_queue = point_cloud_queue # point cloud queue
        self.catch_background = False
        self.background_data = [] 
        self.background_data_process = None
        

        self.zoom = 1.0
        self.offset = np.array([0, 0])
        self.dragging = False
        self.last_mouse_pos = (0, 0)
        self.any_slider_active = False  # Track if any slider is active
        
        # Sliders for parameters
        self.color_intensity_slider = Slider(self.screen, (50, 550, 200, 20), "eps", default_value=0.5)
        self.density_slider = Slider(self.screen, (300, 550, 200, 20), "min_points", default_value=0.5)
        self.switch_bck_recording_mode = ToggleButton(self.screen, (20, 20, 100, 50), 'Idle', 'Generating Backgroud', toggle_catch_background)
        self.switch_tracking_mode = ToggleButton(self.screen, (20, 100, 100, 50), 'Track Off', 'Track On', toggle_tracking_mode)
        self.bck_length_info = InfoBox(self.screen,(650,20,100,50),'No bck info')
        self.gen_bck_bottom = Button(self.screen,(650,100,100,50),'Gen Bck',self.start_background_generation)

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

    def draw(self, data):
        self.screen.fill((0, 0, 0))
        data = (data.T * self.zoom + self.offset[:, None]).T
        color = int(self.color_intensity_slider.value * 255)  # Using the slider value for RGB intensity
        for x, y in data:
            pygame.draw.circle(self.screen, (color,color,color), (x, y), 2)

        self.color_intensity_slider.draw()
        self.density_slider.draw()
        self.switch_bck_recording_mode.draw() 
        self.switch_tracking_mode.draw()
        
        self.bck_length_info.draw()
        self.gen_bck_bottom.draw()
        pygame.display.flip()
    
    
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

    def run(self):
        while self.running:
            self.handle_events()
            self.catch_background
            if not self.point_cloud_queue.empty():
                Td_map = self.point_cloud_queue.get()
                if self.switch_bck_recording_mode.state:
                    self.background_data.append(Td_map)
                    self.bck_length_info.update_text(f"Data Length: {len(self.background_data)}")

                point_cloud_data = get_pcd_uncolored(Td_map)
                density = self.density_slider.value
                ds_point_cloud_data_ind = np.random.choice(np.arange(len(point_cloud_data)), size = int(len(point_cloud_data) * density),replace = False).astype(int)
                point_cloud_data = point_cloud_data[ds_point_cloud_data_ind]
                self.screen.fill((0, 0, 0))  # Clear screen
                self.draw(point_cloud_data)

    def quit(self):
        self.running = False
        if self.background_data_process:
            self.background_data_process.join()  # Wait for the process to complete
        pygame.quit()

if __name__ == '__main__':
    pcap_file_path = r'../../../Data/2019-12-21-7-30-0.pcap'

    set_start_method('fork')
    raw_data_queue = Queue() # Packet Queue
    point_cloud_queue = Queue()
    # Creating processes for Core 2 and Core 3 tasks
    eth_reader = load_pcap(pcap_file_path)

    packet_reader_process = Process(target=read_packets, args=(raw_data_queue,eth_reader,))
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
