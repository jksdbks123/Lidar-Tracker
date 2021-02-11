import open3d as op3
import numpy as np
import os
def save_view_point(pcd, filename):
    vis = op3.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    op3.io.write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()

if __name__ == "__main__":
    json = r'E:\lidar\op3\view_point.json'
    lisdir = os.listdir("E:\lidar\op3\pcds")
    initial_pcd = op3.io.read_point_cloud("E:\lidar\op3\pcds\{}.pcd".format(0))
    save_view_point(initial_pcd,json)
    vis = op3.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([255, 255, 255])
    opt.point_size = 3
    param=op3.io.read_pinhole_camera_parameters(json)
    ctr=vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.get_view_control()
    for i in range(len(lisdir)):
        pcd = op3.io.read_point_cloud("E:\lidar\op3\pcds\{}.pcd".format(i))
        vis.add_geometry(pcd)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.poll_events()
        vis.update_renderer()
        vis.clear_geometries()
    vis.destroy_window()        