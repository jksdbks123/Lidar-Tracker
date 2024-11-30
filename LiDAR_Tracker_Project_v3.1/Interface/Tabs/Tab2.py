from Interface_builder import *

def build_tab2(tab, config, processor):
    """Builds the Batch Processing tab."""
    ttk.Label(tab, text="Batch Processing").grid(column=0, row=0, padx=10, pady=10)

    # Folder and file selection
    batch_folder = StringVar(value=config.get_param("tab2")["batch_folder"])
    output_folder = StringVar(value=config.get_param("tab2")["output_folder"])
    ref_xyz_file = StringVar(value=config.get_param("tab2")["ref_xyz_file_path"])
    ref_llh_file = StringVar(value=config.get_param("tab2")["ref_llh_file_path"])
    diff_to_utc = IntVar(value=config.get_param("tab2")["Diff2UTC"])
    if_save_forepoints = BooleanVar(value=config.get_param("tab2")["SaveForepoints"])
    track_param_vars = dict()
    for key in config.tracking_parameter_dict.keys():
        track_param_vars[key] = DoubleVar(value=config.get_track_param(key))

    ttk.Label(tab, text="Batch Folder").grid(column=0, row=1)
    ttk.Entry(tab, textvariable=batch_folder, width=50).grid(column=1, row=1)
    ttk.Button(
        tab,
        text="Select Folder",
        command=lambda: select_folder("tab2", "batch_folder (.pcap)", batch_folder, config)
    ).grid(column=2, row=1)

    ttk.Label(tab, text="Output Folder").grid(column=0, row=2)
    ttk.Entry(tab, textvariable=output_folder, width=50).grid(column=1, row=2)
    ttk.Button(
        tab,
        text="Select Folder",
        command=lambda: select_folder("tab2", "output_traj_folder (.csv)", output_folder, config)
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
        command=lambda: select_folder("tab2", "point_cloud_folder (.pcd)", point_cloud_folder, config)
    ).grid(column=3, row=7)

    # Buttons
    ttk.Button(
        tab,
        text="Start Batch Processing",
        command=lambda: processor.start_batch_processing(
            batch_folder.get(), output_folder.get(), ref_xyz_file.get(), ref_llh_file.get(), diff_to_utc.get()
        )
    ).grid(column=0, row=6, padx=10, pady=10)
    # create a series of entries in the right side of panel
    #  for following parameters with default values in the dictionary
    for i,key in enumerate(track_param_vars.keys()):
        ttk.Label(tab, text=key).grid(column=0, row=8 + i )
        ttk.Entry(tab, textvariable=track_param_vars[key], width=10).grid(column=1, row=8 + i)