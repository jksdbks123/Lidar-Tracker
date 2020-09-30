from Trakers import kalman_tracker
class TrackingObject():
    def __init__(self,initial_position,initial_point_cloud,initial_bounding_box):
        """
        docstring
        """
        self.tracker = kalman_tracker(initial_position[0],initial_position[1])
        self.detected_centers = [initial_position] # [x,y] narray
        self.estimated_centers = [initial_position] # [x,y] narray
        self.point_clouds = [initial_point_cloud]
        self.bounding_boxes = [initial_bounding_box]
    
class DetectedObject():
    def __init__(self,position,point_cloud,bounding_box):
        self. position = position
        self.point_cloud = point_cloud
        self.bounding_box = bounding_box
    