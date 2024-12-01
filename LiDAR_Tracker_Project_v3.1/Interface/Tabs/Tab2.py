from File_manager import update_flag, select_file, select_folder
from tkinter import ttk
from tkinter import StringVar, IntVar, BooleanVar,DoubleVar
from Tracker import run_batch_mot

def build_tab2(tab, config):
    """Builds the Batch Processing tab."""
    ttk.Label(tab, text="Batch Processing").grid(column=0, row=0, padx=10, pady=10)
    # Folder and file selection
    batch_folder = StringVar(value=config.get_param("tab2")["batch_folder"])
    output_folder = StringVar(value=config.get_param("tab2")["output_folder"])
    ref_xyz_file = StringVar(value=config.get_param("tab2")["ref_xyz_file_path"])
    ref_llh_file = StringVar(value=config.get_param("tab2")["ref_llh_file_path"])
    diff_to_utc = IntVar(value=config.get_param("tab2")["Diff2UTC"])
    win_width = IntVar(value=config.get_track_param("win_width"))
    win_height = IntVar(value=config.get_track_param("win_height"))
    eps = DoubleVar(value=config.get_track_param("eps"))
    min_samples = IntVar(value=config.get_track_param("min_samples"))
    missing_thred = IntVar(value=config.get_track_param("missing_thred"))
    bck_radius = DoubleVar(value=config.get_track_param("bck_radius"))
    N = IntVar(value=config.get_track_param("N"))
    d_thred = DoubleVar(value=config.get_track_param("d_thred"))
    bck_n = IntVar(value=config.get_track_param("bck_n"))
    n_cpu = IntVar(value=config.get_param("tab2")["n_cpu"])

    if_save_forepoints = BooleanVar(value=config.get_param("tab2")["SaveForepoints"])

    ttk.Label(tab, text="Batch Folder").grid(column=0, row=1)
    ttk.Entry(tab, textvariable=batch_folder, width=50).grid(column=1, row=1)
    ttk.Button(
        tab,
        text="Select Folder",
        command=lambda: select_folder("tab2", "batch_folder", batch_folder, config)
    ).grid(column=2, row=1)

    ttk.Label(tab, text="Output Folder").grid(column=0, row=2)
    ttk.Entry(tab, textvariable=output_folder, width=50).grid(column=1, row=2)
    ttk.Button(
        tab,
        text="Select Folder",
        command=lambda: select_folder("tab2", "output_traj_folder", output_folder, config)
    ).grid(column=2, row=2)

    ttk.Label(tab, text="Reference XYZ File").grid(column=0, row=3)
    ttk.Entry(tab, textvariable=ref_xyz_file, width=50).grid(column=1, row=3)
    ttk.Button(
        tab,
        text="Select File",
        command=lambda: select_file("tab2", "ref_xyz_file_path", ref_xyz_file, config, [("CSV files", "*.csv")])
    ).grid(column=2, row=3)

    ttk.Label(tab, text="Reference LLH File").grid(column=0, row=4)
    ttk.Entry(tab, textvariable=ref_llh_file, width=50).grid(column=1, row=4)
    ttk.Button(
        tab,
        text="Select File",
        command=lambda: select_file("tab2", "ref_llh_file_path", ref_llh_file, config, [("CSV files", "*.csv")])
    ).grid(column=2, row=4)

    ttk.Label(tab, text="Diff to UTC").grid(column=0, row=5)
    ttk.Entry(tab, textvariable=diff_to_utc, width=10).grid(column=1, row=5)

    # Checkbox
    ttk.Checkbutton(
        tab,
        text="Enable Point Cloud Saving",
        variable=if_save_forepoints,
        command= lambda: update_flag(if_save_forepoints),
        onvalue=True,
        offvalue=False
    ).grid(column=0, row=7, padx=10, pady=10)
    # Buttons for setting point cloud saving folder 
    point_cloud_folder = StringVar(value=config.get_param("tab2")["point_cloud_folder"])
    ttk.Label(tab, text="Point Cloud Folder").grid(column=1, row=7)
    ttk.Entry(tab, textvariable=point_cloud_folder, width=50).grid(column=2, row=7)
    ttk.Button(
        tab,
        text="Select Folder",
        command=lambda: select_folder("tab2", "point_cloud_folder", point_cloud_folder, config)
    ).grid(column=3, row=7)
    
    # Buttons
    ttk.Button(
        tab,
        text="Start Batch Processing",
        command=lambda: run_batch_mot(
            batch_folder.get(), 
            output_folder.get(),
            point_cloud_folder.get(),
            {'win_width':win_width.get(),'win_height':win_height.get(),
             'eps':eps.get(),'min_samples':min_samples.get(),
             'missing_thred':missing_thred.get(),
             'bck_radius':bck_radius.get(),'N':N.get(),'d_thred':d_thred.get(),'bck_n':bck_n.get()},
            diff_to_utc.get(),
            ref_llh_file.get(),
            ref_xyz_file.get(), 
            n_cpu.get(),
            if_save_forepoints.get(),
            
        )
    ).grid(column=0, row=6, padx=10, pady=10)
    # create a series of entries in the right side of panel
    #  for following parameters with default values in the dictionary
 
    ttk.Label(tab, text="win_width").grid(column=0, row=8)
    ttk.Entry(tab, textvariable=win_width, width=10).grid(column=1, row=8)
    ttk.Label(tab, text="win_height").grid(column=0, row=9)
    ttk.Entry(tab, textvariable=win_height, width=10).grid(column=1, row=9)
    ttk.Label(tab, text="eps").grid(column=0, row=10)
    ttk.Entry(tab, textvariable=eps, width=10).grid(column=1, row=10)
    ttk.Label(tab, text="min_samples").grid(column=0, row=11)
    ttk.Entry(tab, textvariable=min_samples, width=10).grid(column=1, row=11)
    ttk.Label(tab, text="missing_thred").grid(column=0, row=12)
    ttk.Entry(tab, textvariable=missing_thred, width=10).grid(column=1, row=12)
    ttk.Label(tab, text="bck_radius").grid(column=0, row=13)
    ttk.Entry(tab, textvariable=bck_radius, width=10).grid(column=1, row=13)
    ttk.Label(tab, text="N").grid(column=0, row=14)
    ttk.Entry(tab, textvariable=N, width=10).grid(column=1, row=14)
    ttk.Label(tab, text="d_thred").grid(column=0, row=15)
    ttk.Entry(tab, textvariable=d_thred, width=10).grid(column=1, row=15)
    ttk.Label(tab, text="bck_n").grid(column=0, row=16)
    ttk.Entry(tab, textvariable=bck_n, width=10).grid(column=1, row=16)
    ttk.Label(tab, text="n_cpu").grid(column=0, row=17)
    ttk.Entry(tab, textvariable=n_cpu, width=10).grid(column=1, row=17)
    

        
        
