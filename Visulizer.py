import open3d as op3
import numpy as np
import os
import time 
import argparse
def save_view_point(Initial_pcd,PCD_folder):
    
    vis = op3.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(Initial_pcd)
    op3.visualization
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    #./Output File/view_point.json 
    filename = os.path.join(PCD_folder,'view_point.json')
    op3.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()
    
    return filename
    
def show_3d_sequence(PCD_folder):
    
    lisdir = os.listdir(PCD_folder)
    lisdir = np.array([l for l in lisdir if l.split('.')[1] == 'pcd'])
    # exclude file path that is not .pcd
    inds = np.array([l[:-4] for l in lisdir]).astype('int')
    lisdir = lisdir[np.argsort(inds)]
    #initial pcd
    pcd_name = os.path.join(PCD_folder,lisdir[0])
    initial_pcd = op3.io.read_point_cloud(pcd_name)
    veiw_control_path = save_view_point(initial_pcd,PCD_folder)
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
    
    for i in range(0,len(lisdir)):
        time.sleep(0.1)
        print(i)
        pcd_name = os.path.join(PCD_folder,lisdir[i])
        pcd = op3.io.read_point_cloud(pcd_name)
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        vis.clear_geometries()

# if __name__ == "__main__":
    
    # parser = argparse.ArgumentParser(description='This is a program visualize sequence .pcd files')
    # parser.add_argument('-i','--input', help='folder path that contains .pcd files', required=True)
    # args = parser.parse_args()
    # show_3d_sequence(args.input) 
    
print(os.getcwd())
input_path = '../RawLidarData/McCarranEvans_Test/OutputFile/OutputPcd'
show_3d_sequence(input_path) 
##ss
