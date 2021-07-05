import open3d as op3
import numpy as np
import os
import time 
def save_view_point(initial_pcd):
    vis = op3.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(initial_pcd)
    op3.visualization
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    #./Output File/view_point.json 
    filename = os.path.join(os.path.join(os.getcwd(),'Output File'),'view_point.json')
    op3.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()
    return filename
    
def show_3d_sequence():
    pcds_path = os.path.join(os.path.join(os.getcwd(),'Output File'),'Output Pcd')
    initial_pcd = op3.io.read_point_cloud(pcds_path+"/{}.pcd".format(500))
    veiw_control_path = save_view_point(initial_pcd)
    vis = op3.visualization.Visualizer()
    vis.create_window()
    
    #animation settings 
    opt = vis.get_render_option()
    opt.background_color = np.asarray([255, 255, 255])
    opt.point_size = 3
    
    param=op3.io.read_pinhole_camera_parameters(veiw_control_path)
    ctr=vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)
    
    vis.add_geometry(initial_pcd)
    
    lisdir = os.listdir((pcds_path))
    for i in range(0,len(lisdir)):
        time.sleep(0.1)
        pcd = op3.io.read_point_cloud(pcds_path+"/{}.pcd".format(i))
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        vis.clear_geometries()
    vis.destroy_window()  

if __name__ == "__main__":
    os.chdir(r'/Users/czhui960/Documents/Lidar/RawLidarData/FrameSamplingTest')
    # show_3d_sequence()
    print(os.listdir(r'./Output File/Output Pcd'))