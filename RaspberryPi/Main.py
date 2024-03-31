from multiprocessing import Process, Queue, Event, Manager
import time
from multiprocessing import set_start_method
from Utils import *
from GenBckFile import *
from LiDARBase import *
from MOT_TD_BCKONLIONE import MOT

class LidarVisualizer:
    def __init__(self,point_cloud_queue, tracking_result_queue,raw_data_queue,tracking_parameter_dict,tracking_param_update_event,width=1500, height=1000, title='LiDAR Data Visualization'):
        """
        Coordinate Logic:
        1) Mouse Coord 2) World Coord 3) Screen Coord

        Mouse -> Y-axis upside down -> Screen Coord -> offset and zoom -> World

        """
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        self.background_surface = pygame.Surface((width, height))
        self.background_surface.fill((0, 0, 0))  # Fill with black or your choice of background color
        self.vertical_limits = [0,31] # Vertical range (for 32 lines)
        pygame.display.set_caption(title)
        pygame.font.init()  # Initialize the font module
        self.object_label_font = pygame.font.SysFont('Comic Sans MS', 20)
        self.running = True
        self.point_cloud_queue = point_cloud_queue # point cloud queue
        self.tracking_result_queue = tracking_result_queue
        self.raw_data_queue = raw_data_queue
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
        self.static_bck_points = None

        if os.path.exists(r'./thred_map.npy'):
            self.thred_map = np.load(r'./thred_map.npy')
            self.static_bck_points = get_static_bck_points(self.thred_map,self.vertical_limits)
            print("Background loaded")
        
        self.if_background_need_update = False 

        self.mot = None
        self.zoom = 1.0
        self.offset = np.array([0, 0])
        self.dragging = False
        self.last_mouse_pos = (0, 0)

        # adjusted_points = adjust_for_zoom_and_offset_numpy(self.static_bck_points,self.zoom,self.offset)
        for point in self.static_bck_points:
            x,y = point
            x,y = self.convert_coordinates(x,y)
            pygame.draw.circle(self.background_surface, (255,255,255), (x,y), 2)
        self.screen.blit(self.background_surface, (0, 0))

        self.any_slider_active = False  # Track if any slider is active
        self.bar_drawer = BarDrawer() # bar crossing for vehicle counting
        self.lane_drawer = LaneDrawer() # lane drawer for queue detection
        self.lane_drawer.update_lane_gdf()         

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
        self.switch_queue_monitoring_mode = ToggleButton(self.screen, (20, 200, 100, 30), 'Queue Monitor Off', 'Queue Monitor On', self.toggle_queue_monitor)
        # left middle
        self.switch_drawing_lines_mode = ToggleButton(self.screen, (20, 550, 100, 30), 'Bar Edit Off', 'Bar Edit On',self.draw_lines)
        self.buttom_clear_lines = Button(self.screen,(20,600,100,30),'Clear Lines',self.bar_drawer.clear)

        self.switch_drawing_lanes_mode = ToggleButton(self.screen, (20, 650, 100, 30), 'Lane Edit Off', 'Lane Edit On',self.draw_lanes)
        self.buttom_clear_lanes = Button(self.screen,(20,700,100,30),'Clear Lanes',self.lane_drawer.clear)
        self.lane_width_info = InfoBox(self.screen,(20,750,100,30),'Lane Width: 12ft')

        # right upper
        self.bck_length_info = InfoBox(self.screen,(1350,20,100,30),'No bck info')
        self.gen_bck_bottom = Button(self.screen,(1350,80,100,30),'Gen Bck',self.start_background_generation)

        # bottom 
        self.density_slider = Slider(self.screen, (1200, 900, 200, 20), "PC density",0,1,default_value=1)
        self.frame_process_time_info = InfoBox(self.screen,(1200,840,250,30),'0ms')
        self.db_window_width_slider =  Slider(self.screen, (50, 900, 200, 20), "eps_width",2,30, default_value=0.4, if_int = True, if_odd = True)
        self.db_window_height_slider =  Slider(self.screen, (300, 900, 200, 20), "eps_height",2,30, default_value=0.2, if_int = True, if_odd = True)
        self.db_min_samples_slider =  Slider(self.screen, (50, 840, 200, 20), "min_samples",2,50, default_value=0.1, if_int = True)
        self.db_eps_dis_slider =  Slider(self.screen, (300, 840, 200, 20), "eps_dis",0,5, default_value=0.2)
        self.update_tracking_param_buttom = Button(self.screen,(550,900,100,30),'Update Param',self.update_tracking_param)
        
        self.info_boxes = [self.bck_length_info,self.frame_process_time_info,self.lane_width_info]
        self.slider_bars = [self.db_window_width_slider,self.db_window_height_slider,self.db_min_samples_slider,self.db_eps_dis_slider,self.density_slider]
        self.events_handle_items = [self.switch_bck_recording_mode,self.switch_tracking_mode,self.switch_foreground_mode,self.switch_object_id,self.update_tracking_param_buttom,
                                    self.gen_bck_bottom,self.switch_drawing_lines_mode,self.buttom_clear_lines,self.buttom_clear_lanes,self.switch_drawing_lanes_mode,self.switch_queue_monitoring_mode] # buttoms and toggles
        self.toggle_buttons = [self.switch_bck_recording_mode,self.switch_foreground_mode,self.switch_tracking_mode,self.switch_drawing_lanes_mode,self.switch_drawing_lines_mode,self.switch_queue_monitoring_mode] 
        # Background parameters
        self.bck_radius = 0.2

    def convert_coordinates(self, x, y):
        """Converts y-coordinate to simulate origin at bottom-left."""
        return x, self.screen.get_height()  - y

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
            if not self.any_slider_active and not self.bar_drawer.drawing_lines and not self.lane_drawer.drawing_lanes:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button for dragging
                        self.dragging = True
                        mouse_pos = event.pos
                        mouse_pos = self.convert_coordinates(mouse_pos[0],mouse_pos[1])
                        self.last_mouse_pos = mouse_pos
                    elif event.button == 4:  # Mouse wheel up to zoom in
                        self.zoom *= 1.1
                        self.if_background_need_update = True
                    elif event.button == 5:  # Mouse wheel down to zoom out
                        self.zoom /= 1.1
                        self.if_background_need_update = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Stop dragging on left mouse button release
                        self.dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    if self.dragging:
                        self.if_background_need_update = True
                        mouse_pos = event.pos
                        mouse_pos = self.convert_coordinates(mouse_pos[0],mouse_pos[1])
                        movement = np.array(mouse_pos) - np.array(self.last_mouse_pos)
                        self.offset += movement
                        self.last_mouse_pos = mouse_pos

            if self.bar_drawer.drawing_lines:
                
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    hover_flag = True
                    for badget in self.events_handle_items:
                        hover_flag = hover_flag and not badget.is_mouse_over(event.pos)
                    if hover_flag:
                        if not self.bar_drawer.start_drawing_lines:
                            self.bar_drawer.start_drawing_lines = True
                            mouse_pos = event.pos
                            mouse_pos = self.convert_coordinates(mouse_pos[0],mouse_pos[1])
                            world_x = (mouse_pos[0] - self.offset[0]) / self.zoom
                            world_y = (mouse_pos[1] - self.offset[1]) / self.zoom
                            self.bar_drawer.current_line_start = (world_x,world_y)
                        else:
                            # Finish drawing the line
                            mouse_pos = event.pos
                            mouse_pos = self.convert_coordinates(mouse_pos[0],mouse_pos[1])
                            world_x = (mouse_pos[0] - self.offset[0]) / self.zoom
                            world_y = (mouse_pos[1] - self.offset[1]) / self.zoom
                            self.bar_drawer.lines.append((self.bar_drawer.current_line_start, (world_x,world_y)))
                            self.bar_drawer.line_counts.append(0)
                            self.bar_drawer.start_drawing_lines = False
                            self.bar_drawer.current_line_start = None
                            self.bar_drawer.save()
                
                if self.bar_drawer.start_drawing_lines and event.type == pygame.MOUSEMOTION:
                    mouse_pos = event.pos
                    mouse_pos = self.convert_coordinates(mouse_pos[0],mouse_pos[1])
                    world_x = (mouse_pos[0] - self.offset[0]) / self.zoom
                    world_y = (mouse_pos[1] - self.offset[1]) / self.zoom
                    self.bar_drawer.current_line_connection = (self.bar_drawer.current_line_start,(world_x, world_y))
            """
            Left click to create a polyline
            Right click to discard current step
            Middle click to finish a polyline drawing
            """

            if self.lane_drawer.drawing_lanes:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_MINUS:
                        self.lane_drawer.lane_width = max(0.3048, self.lane_drawer.lane_width - 0.3048)
                        self.lane_width_info.update_text(f'Lane Width: {self.lane_drawer.lane_width / 0.3048:.1f}ft')
                    if event.key == pygame.K_EQUALS:
                        self.lane_drawer.lane_width += 0.3048
                        self.lane_width_info.update_text(f'Lane Width: {self.lane_drawer.lane_width / 0.3048:.1f}ft')
                    if event.key == pygame.K_d:
                        self.lane_drawer.remove_last_record()
                            
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1: # add spline 
                    hover_flag = True
                    for badget in self.events_handle_items:
                        hover_flag = hover_flag and not badget.is_mouse_over(event.pos)
                
                    if hover_flag: # not hover on the toggle or buttoms
                        if not self.lane_drawer.start_drawing_lanes:
                            self.lane_drawer.start_drawing_lanes = True
                            mouse_pos = event.pos
                            mouse_pos = self.convert_coordinates(mouse_pos[0],mouse_pos[1])
                            world_x = (mouse_pos[0] - self.offset[0]) / self.zoom
                            world_y = (mouse_pos[1] - self.offset[1]) / self.zoom
                            self.lane_drawer.current_lane_points.append((world_x,world_y))

                        else:
                            # continue add spline points into the list
                            mouse_pos = event.pos
                            mouse_pos = self.convert_coordinates(mouse_pos[0],mouse_pos[1])
                            world_x = (mouse_pos[0] - self.offset[0]) / self.zoom
                            world_y = (mouse_pos[1] - self.offset[1]) / self.zoom
                            self.lane_drawer.current_lane_points.append((world_x,world_y))
                            self.lane_drawer.current_lane_widths.append(self.lane_drawer.lane_width)

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3: # withdraw last point
                    # if we are drawing and the list is empty, then pop the last one
                    if self.lane_drawer.start_drawing_lanes and self.lane_drawer.current_lane_points:
                        self.lane_drawer.current_lane_points.pop() # remove the last one
                        if self.lane_drawer.current_lane_widths:
                            self.lane_drawer.current_lane_widths.pop()
                            

                    if not self.lane_drawer.current_lane_points:
                        # if it's empty after we pop out the last one, then quit drawing session
                        self.lane_drawer.start_drawing_lanes = False
                        self.lane_drawer.current_lane_connection = None

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 2: # middle click
                    if len(self.lane_drawer.current_lane_points) > 1:
                        self.lane_drawer.lane_end_points.append(self.lane_drawer.current_lane_points[-1])
                        lane_multipoly = create_subsection_polygons(self.lane_drawer.current_lane_points,self.lane_drawer.current_lane_widths,0.5)
                        self.lane_drawer.lane_subsections_poly.append(lane_multipoly)
                        lane_vertices = []
                        for poly in lane_multipoly:
                            lane_vertices.append(list(poly.exterior.coords))
                        self.lane_drawer.lane_points.append(lane_vertices)
                        self.lane_drawer.start_drawing_lanes = False
                        self.lane_drawer.current_lane_connection = None
                        self.lane_drawer.current_lane_points = []  # List of points defining the current lane centerline
                        self.lane_drawer.current_lane_widths = []
                        self.lane_drawer.save()
                        self.lane_drawer.update_lane_gdf()   

                if self.lane_drawer.start_drawing_lanes and event.type == pygame.MOUSEMOTION and self.lane_drawer.current_lane_points:
                    mouse_pos = event.pos
                    mouse_pos = self.convert_coordinates(mouse_pos[0],mouse_pos[1])
                    world_x = (mouse_pos[0] - self.offset[0]) / self.zoom
                    world_y = (mouse_pos[1] - self.offset[1]) / self.zoom
                    self.lane_drawer.current_lane_connection = (self.lane_drawer.current_lane_points[-1],(world_x, world_y))

                            
    def draw_manual_elements(self,point_data,point_label):
        """
        Current Drawing Sessions
        """
        if self.lane_drawer.start_drawing_lanes and self.lane_drawer.current_lane_connection is not None:
            
            his_cur_centerline = self.lane_drawer.current_lane_points + [self.lane_drawer.current_lane_connection[1]]
            his_cur_width = self.lane_drawer.current_lane_widths + [self.lane_drawer.lane_width]
            his_poly = create_bufferzone_vertex(his_cur_centerline,his_cur_width)
            his_poly = adjust_for_zoom_and_offset(his_poly,self.zoom,self.offset)
            his_poly_converted = []
            for x,y in his_poly:
                his_poly_converted.append(self.convert_coordinates(x,y))
            pygame.draw.polygon(self.screen, (0, 255, 0), his_poly_converted)

        if self.bar_drawer.start_drawing_lines and self.bar_drawer.current_line_connection is not None:
            
            start_x, start_y = self.bar_drawer.current_line_connection[0]
            end_x, end_y = self.bar_drawer.current_line_connection[1]
            adjusted_start_x = (start_x * self.zoom) + self.offset[0]
            adjusted_start_y = (start_y * self.zoom) + self.offset[1]
            adjusted_end_x = (end_x * self.zoom) + self.offset[0]
            adjusted_end_y = (end_y * self.zoom) + self.offset[1]
            adjusted_start_x,adjusted_start_y = self.convert_coordinates(adjusted_start_x,adjusted_start_y)
            adjusted_end_x,adjusted_end_y = self.convert_coordinates(adjusted_end_x,adjusted_end_y)
            pygame.draw.line(self.screen, (122, 128, 214), (adjusted_start_x, adjusted_start_y), (adjusted_end_x, adjusted_end_y), 5)
        
        """
        Historical Drawing Sessions
        """
        for i,line in enumerate(self.bar_drawer.lines):

            count = self.bar_drawer.line_counts[i]
            start_x, start_y = line[0]
            end_x, end_y = line[1]
            
            adjusted_start_x = (start_x * self.zoom) + self.offset[0]
            adjusted_start_y = (start_y * self.zoom) + self.offset[1]
            adjusted_end_x = (end_x * self.zoom) + self.offset[0]
            adjusted_end_y = (end_y * self.zoom) + self.offset[1]
            adjusted_start_x,adjusted_start_y = self.convert_coordinates(adjusted_start_x,adjusted_start_y)
            adjusted_end_x,adjusted_end_y = self.convert_coordinates(adjusted_end_x,adjusted_end_y)
            # Draw the line
            pygame.draw.line(self.screen, (122, 128, 214), (adjusted_start_x, adjusted_start_y), (adjusted_end_x, adjusted_end_y), 5)
            
            # Calculate midpoint for the count text, adjusted for zoom and offset
            mid_point_x = ((adjusted_start_x + adjusted_end_x) / 2)
            mid_point_y = ((adjusted_start_y + adjusted_end_y) / 2)
            # Render the count text
            count_surf = self.object_label_font.render(f'id{i}:{count}', True, (200, 128, 20))
            self.screen.blit(count_surf, (mid_point_x - count_surf.get_width() / 2, mid_point_y - count_surf.get_height() / 2))
        
        if self.lane_drawer.lane_points:
            if self.switch_queue_monitoring_mode.state:
                lane_section_foreground_point_counts = get_lane_section_foreground_point_counts(self.lane_drawer.lane_subsections_poly,
                                                                                                self.lane_drawer.lane_gdf,
                                                                                                point_data,point_label)
                self.lane_drawer.lane_section_foreground_point_counts = lane_section_foreground_point_counts

            for i,lane_poly_points in enumerate(self.lane_drawer.lane_points):
                for j,seg_poly_points in enumerate(lane_poly_points):
                    poly = adjust_for_zoom_and_offset(seg_poly_points,self.zoom,self.offset)
                    poly_converted = []
                    for x,y in poly:
                        poly_converted.append(self.convert_coordinates(x,y))
                    color = (0, 255, 0)
                    if self.switch_queue_monitoring_mode.state:
                        if self.lane_drawer.lane_section_foreground_point_counts[i][j] > 10:
                            color = (255, 0, 0) # red
                    pygame.draw.polygon(self.screen, color, poly_converted)

                label_surface = self.object_label_font.render(f'Lane {i}', False, (255,65,212))
                x,y = self.lane_drawer.lane_end_points[i]
                label_pos = (x * self.zoom + self.offset[0],y * self.zoom + self.offset[1])
                label_pos_x,label_pos_y = self.convert_coordinates(label_pos[0],label_pos[1])
                self.screen.blit(label_surface,(label_pos_x,label_pos_y))
            

    def draw(self, data_raw, point_label = None, tracking_dic = None):
        self.screen.fill((0, 0, 0))
        # data = (data.T * self.zoom + self.offset[:, None]).T # nx2 
        data = adjust_for_zoom_and_offset_numpy(data_raw,self.zoom,self.offset)

        if self.switch_foreground_mode.state:
            for coord,l in zip(data,point_label):
                x,y = coord
                # x,y = self.convert_coordinates(x,y)
                x,y = self.convert_coordinates(x,y)
                pygame.draw.circle(self.screen, tuple(color_map[l]), (x, y), 2)

        elif self.switch_bck_recording_mode.state:
            for x, y in data:
                x,y = self.convert_coordinates(x,y)
                pygame.draw.circle(self.screen, (255,255,255), (x, y), 2)
        else:
            if self.if_background_need_update and self.thred_map is not None:
                self.background_surface.fill((0,0,0))
                static_bck_points_adjusted = adjust_for_zoom_and_offset_numpy(self.static_bck_points,self.zoom,self.offset)
                for point in static_bck_points_adjusted:
                    x,y = point
                    x,y = self.convert_coordinates(x,y)
                    pygame.draw.circle(self.background_surface, (255,255,255), (x, y), 2)
                self.if_background_need_update = False
            self.screen.blit(self.background_surface, (0, 0))
            self.draw_manual_elements(data_raw,point_label)
            
            if self.switch_tracking_mode.state:
                # point_label is correspondance to the obj_id
                data = data[point_label != -1]
                point_label = point_label[point_label != -1]
                for obj_id in tracking_dic.keys():
                    color_vec = color_map[obj_id%len(color_map)]
                    if self.if_objid:
                        cur_traj = tracking_dic[obj_id].mea_seq[-1] # -1 happens here sometimes
                        if cur_traj is not None:
                            label_surface = self.object_label_font.render(str(obj_id), False, (255,65,212))
                            x,y = (cur_traj[0][0][0] + cur_traj[1][0][0])/2 , (cur_traj[0][1][0] + cur_traj[1][1][0])/2
                            label_pos = (x * self.zoom + self.offset[0],y * self.zoom + self.offset[1])
                            self.screen.blit(label_surface,label_pos)

                    for coord in data[point_label == obj_id]:
                        x,y = coord
                        x,y = self.convert_coordinates(x,y)
                        pygame.draw.circle(self.screen, color_vec, (x, y), 2)

                    his_coords = tracking_dic[obj_id].mea_seq[-10:]
                    for coord in his_coords:
                        if coord is not None:
                            x = (coord[0][0][0] + coord[1][0][0]) / 2
                            y = (coord[0][1][0] + coord[1][1][0]) / 2
                            x = x * self.zoom +  self.offset[0]
                            y = y * self.zoom +  self.offset[1]
                            x,y = self.convert_coordinates(x,y)
                            pygame.draw.circle(self.screen, tuple(color_vec), (x, y), 4)

                    if len(tracking_dic[obj_id].post_seq) > 1:
                        prev_pos = tracking_dic[obj_id].post_seq[-2][0].flatten()[:2]
                        curr_pos = tracking_dic[obj_id].post_seq[-1][0].flatten()[:2]
                        for i in range(len(self.bar_drawer.line_counts)):
                            if line_segments_intersect(prev_pos, curr_pos, self.bar_drawer.lines[i][0], self.bar_drawer.lines[i][1]):
                                self.bar_drawer.line_counts[i] += 1
                                break
            else:
                
                for coord,l in zip(data,point_label):
                    if l == 0:
                        continue
                    x,y = coord
                    x,y = self.convert_coordinates(x,y)
                    pygame.draw.circle(self.screen, tuple(color_map[1]), (x, y), 2)

        for item in self.events_handle_items:
            item.draw()
        for item in self.info_boxes:
            item.draw()
        for item in self.slider_bars:
            item.draw() 

        pygame.display.flip()

    def draw_lines(self,state):

        if state:
            self.bar_drawer.drawing_lines = True
            self.deactivate_other_toggles(self.switch_drawing_lines_mode)
        else:
            self.bar_drawer.drawing_lines = False
            self.bar_drawer.start_drawing_lines =  False
            self.bar_drawer.current_line_start = None

    def draw_lanes(self,state):
        if state:
            self.lane_drawer.drawing_lanes = True
            self.deactivate_other_toggles(self.switch_drawing_lanes_mode)
        else:
            self.lane_drawer.drawing_lanes = False
            self.lane_drawer.start_drawing_lanes =  False
            self.lane_drawer.current_lane_connection = None


    def deactivate_other_toggles(self,activate_button):
        for button in self.toggle_buttons:
            if button != activate_button:
                button.set_state()

    def toggle_catch_background(self,state):
        if state:
            self.deactivate_other_toggles(self.switch_bck_recording_mode)
        print('Test_toggle_bck')

    def toggle_queue_monitor(self,state):
        if state:
            self.deactivate_other_toggles(self.switch_queue_monitoring_mode)
            
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
                self.tracking_prcess = Process(target=track_point_clouds, args=(self.tracking_process_stop_event,self.mot,self.point_cloud_queue,self.tracking_result_queue,self.tracking_parameter_dict,self.tracking_param_update_event,))
                self.tracking_prcess.start()
        else:
            if self.tracking_prcess and self.tracking_prcess.is_alive():
                self.mot = None
                self.tracking_process_stop_event.set()
                self.tracking_prcess.join()
                self.tracking_prcess = None
                while not self.tracking_result_queue.empty():
                    self.tracking_result_queue.get_nowait()
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

    def update_background(self):
        # Check if the background process has finished and thred_map is not loaded yet
        if self.background_data_process and not self.background_data_process.is_alive() and not self.thred_map_loaded:
            self.thred_map = np.load(r'./thred_map.npy')
            self.thred_map_loaded = True
            print("thred_map.npy loaded")
            self.static_bck_points = get_static_bck_points(self.thred_map,self.vertical_limits)
    
    def run(self):

        while self.running:
            
            self.handle_events()
            time_cums = 0

            
            if self.switch_bck_recording_mode.state:
                Td_map = self.point_cloud_queue.get()
                self.background_data.append(Td_map)
                self.bck_length_info.update_text(f"Data Length: {len(self.background_data)}")
                point_cloud_data,point_labels,tracking_dic = get_ordinary_point_cloud(Td_map,self.vertical_limits)
                # pc,None,None
            elif self.switch_foreground_mode.state:
                Td_map = self.point_cloud_queue.get()
                point_cloud_data,point_labels,tracking_dic = get_foreground_point_cloud(self.thred_map,self.bck_radius,
                                                                                        Td_map,self.vertical_limits)
                # pc,label (0/1),None 
            elif self.switch_tracking_mode.state:
                Tracking_pool,Labeling_map,Td_map,time_cums = self.tracking_result_queue.get()
                point_cloud_data,point_labels = get_pcd_tracking(Td_map,Labeling_map,Tracking_pool,self.vertical_limits)
                # pc, label (obj_id)
                tracking_dic = Tracking_pool

            # elif self.switch_queue_monitoring_mode.state:
            #     Td_map = self.point_cloud_queue.get()
            #     point_cloud_data,point_labels,tracking_dic = get_ordinary_point_cloud(Td_map,self.vertical_limits) 
            else: # default
                Td_map = self.point_cloud_queue.get()
                
                if self.thred_map is not None:
                    point_cloud_data,point_labels,tracking_dic = get_foreground_point_cloud(self.thred_map,self.bck_radius,
                                                                                            Td_map,self.vertical_limits)
                    # pc,label (0/1), None
                else:
                    
                    point_cloud_data,point_labels,tracking_dic = get_ordinary_point_cloud(Td_map,self.vertical_limits)
                    # pc,None, None

            self.update_background() # This is for update background map itself
            time_a = time.time()
            self.screen.fill((0, 0, 0))
            self.draw(point_cloud_data,point_labels,tracking_dic)
            self.frame_process_time_info.update_text(f'resq:{self.tracking_result_queue.qsize()},pcq:{self.point_cloud_queue.qsize()},render:{(time.time() - time_a)*1000:.1f}ms,tracking:{time_cums * 1000:.1f}')
            
                
                      
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

def main(mode = 'online',pcap_file_path = None):
    # pcap_file_path = r'/Users/zhihiuchen/Documents/Data/2019-12-21-7-30-0.pcap'
    
    try:
        with Manager() as manger:
            # set_start_method('fork',force=True)
            raw_data_queue = manger.Queue() # Packet Queue
            point_cloud_queue = manger.Queue()
            tracking_result_queue = manger.Queue() # this is for the tracking results (pt,...)
            tracking_parameter_dict = manger.dict({})
            tracking_param_update_event = Event()
            # Creating processes for Core 2 and Core 3 tasks
            if mode == 'online':
                sock = socket.socket(socket.AF_INET, # Internet
                                socket.SOCK_DGRAM) # UDP
                sock.bind(('', 2368)) 
                packet_reader_process = Process(target=read_packets_online, args=(sock,raw_data_queue,))
            elif mode == 'offline':
                packet_reader_process = Process(target=read_packets_offline, args=(raw_data_queue,pcap_file_path,))

            packet_parser_process = Process(target=parse_packets, args=(raw_data_queue, point_cloud_queue,))
            packet_reader_process.start()
            packet_parser_process.start()

            # Running the visualization (Core 1 task) in the main process
            
            visualizer = LidarVisualizer(point_cloud_queue,tracking_result_queue,raw_data_queue,tracking_parameter_dict,tracking_param_update_event,)
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
        if visualizer.tracking_prcess is not None:
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
    pcap_file_path = r'../../../Data/9thVir/2024-03-14-23-30-00.pcap'
    mode = 'offline'
    main(mode=mode,pcap_file_path = pcap_file_path)
    # mode = 'online'
    # main(mode=mode)