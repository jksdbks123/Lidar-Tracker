import pandas as pd
from tkinter import filedialog
from tkinter import StringVar, ttk, IntVar
from File_manager import update_flag, select_file, select_folder


def build_tab6(tab, config):
    "Count Number of Pedestrians in Pedestrian Detections"
    ttk.Label(tab, text="Count Pedestrian").grid(column=0, row=0, padx=10, pady=10)

    # Variables for dynamic column specification
    traj_folder = StringVar(value=config.get_param("tab6")["trajectory_folder"])
    output_folder = StringVar(value=config.get_param("tab6")["output_folder"])
    point_number_name_column = StringVar(value=config.get_param("tab6")['default_point_number_col_name'])  # Default column name for pcap_name
    area_name_column = StringVar(value=config.get_param("tab6")['default_area_col_name'])  # Default column name for frame_index
    distance_name_column = StringVar(value=config.get_param("tab6")['default_distance_col_name'])  # Default column name for output_name
    n_cpu = IntVar(value=config.get_param("tab6")['n_cpu'])

        # File and folder selection
    ttk.Label(tab, text="Trajectory Folder (.csv)").grid(column=0, row=1)
    ttk.Entry(tab, textvariable=traj_folder, width=50).grid(column=1, row=1)
    ttk.Button(
        tab,
        text="Select Folder",
        command=lambda: select_folder("tab6", "trajectory_folder", traj_folder, config)
    ).grid(column=2, row=1)

    ttk.Label(tab, text="Output Folder").grid(column=0, row=2)
    ttk.Entry(tab, textvariable=output_folder, width=50).grid(column=1, row=2)
    ttk.Button(
        tab,
        text="Select Folder",
        command=lambda: select_folder("tab6", "output_folder", output_folder, config)
    ).grid(column=2, row=2)

    # Dynamic column name specification
    ttk.Label(tab, text="Point Number Column").grid(column=0, row=4)
    ttk.Entry(tab, textvariable=point_number_name_column, width=50).grid(column=1, row=4)

    ttk.Label(tab, text="Area Column").grid(column=0, row=5)
    ttk.Entry(tab, textvariable=area_name_column, width=50).grid(column=1, row=5)

    ttk.Label(tab, text="Distance Column").grid(column=0, row=6)
    ttk.Entry(tab, textvariable=distance_name_column, width=10).grid(column=1, row=6)

    ttk.Label(tab, text="CPU #").grid(column=2, row=6)
    ttk.Entry(tab, textvariable=n_cpu, width=10).grid(column=3, row=6)

    # Start button
    ttk.Button(
        tab,
        text="Start Pedestrian Counting",
        command=lambda: run_batch_clipping_threaded(
            pcap_folder.get(),
            output_folder.get(),
            time_reference_file.get(),
            date_column.get(),
            frame_column.get(),
            time_interval.get(),
            output_name_column.get(),
            n_cpu.get()
        )
    ).grid(column=0, row=8, padx=10, pady=10)