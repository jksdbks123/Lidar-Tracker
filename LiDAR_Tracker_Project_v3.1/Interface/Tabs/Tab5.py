import pandas as pd
from tkinter import filedialog
from tkinter import StringVar, ttk, IntVar
from File_manager import update_flag, select_file, select_folder
from VideoClipping import *

def build_tab5(tab, config):
    """Builds the Video Clipping tab."""
    ttk.Label(tab, text="Video Clipping").grid(column=0, row=0, padx=10, pady=10)

    # Variables for dynamic column specification
    video_folder = StringVar(value=config.get_param("tab5")["video_folder"])
    output_folder = StringVar(value=config.get_param("tab5")["output_folder"])
    time_reference_file = StringVar(value=config.get_param("tab5")["time_reference_file_path"])
    date_column = StringVar(value=config.get_param("tab5")['default_datetime_col_name'])  # Default column name for pcap_name
    frame_column = StringVar(value=config.get_param("tab5")['default_frame_index_col_name'])  # Default column name for frame_index
    output_name_column = StringVar(value=config.get_param("tab5")['output_naming_col_name'])  # Default column name for output_name
    time_interval = IntVar(value=config.get_param("tab5")['time_interval'])
    n_cpu = IntVar(value=config.get_param("tab5")['n_cpu'])
    # File and folder selection
    ttk.Label(tab, text="Video Folder").grid(column=0, row=1)
    ttk.Entry(tab, textvariable=video_folder, width=50).grid(column=1, row=1)
    ttk.Button(
        tab,
        text="Select Folder",
        command=lambda: select_folder("tab5", "video_folder", video_folder, config)
    ).grid(column=2, row=1)

    ttk.Label(tab, text="Output Folder").grid(column=0, row=2)
    ttk.Entry(tab, textvariable=output_folder, width=50).grid(column=1, row=2)
    ttk.Button(
        tab,
        text="Select Folder",
        command=lambda: select_folder("tab5", "output_folder", output_folder, config)
    ).grid(column=2, row=2)

    ttk.Label(tab, text="Time Reference File").grid(column=0, row=3)
    ttk.Entry(tab, textvariable=time_reference_file, width=50).grid(column=1, row=3)
    ttk.Button(
        tab,
        text="Select File",
        command=lambda: select_file("tab5", "time_reference_file", time_reference_file, config, [("CSV files", "*.csv")])
    ).grid(column=2, row=3)

    # Dynamic column name specification
    ttk.Label(tab, text="Date Name Column").grid(column=0, row=4)
    ttk.Entry(tab, textvariable=date_column, width=50).grid(column=1, row=4)

    ttk.Label(tab, text="Frame Index Column").grid(column=0, row=5)
    ttk.Entry(tab, textvariable=frame_column, width=50).grid(column=1, row=5)

    ttk.Label(tab, text="Time Interval (s)").grid(column=0, row=6)
    ttk.Entry(tab, textvariable=time_interval, width=10).grid(column=1, row=6)
    ttk.Label(tab, text="CPU #").grid(column=2, row=6)
    ttk.Entry(tab, textvariable=n_cpu, width=10).grid(column=3, row=6)

    ttk.Label(tab, text="Output Name Column").grid(column=0, row=7)
    ttk.Entry(tab, textvariable=output_name_column, width=50).grid(column=1, row=7)

    # Start button
    ttk.Button(
        tab,
        text="Start PCAP Clipping",
        command=lambda: run_batch_video_clipping_threaded(
            video_folder.get(),
            output_folder.get(),
            time_reference_file.get(),
            date_column.get(),
            frame_column.get(),
            time_interval.get(),
            output_name_column.get(),
            n_cpu.get()
        )
    ).grid(column=0, row=8, padx=10, pady=10)
