import os
import sys
# print script name and system path
from tkinter import ttk
from tkinter import StringVar, IntVar, BooleanVar
from tkinter import filedialog
from tkinter.messagebox import showinfo
from Utils import ExamPcapStartTime
def update_flag(flag):
    pass
    
def build_interface(tabs, config, processor, visualizer, dummy_processor):
    """
    Build the GUI interface for the tabs.

    Args:
        tabs (dict): Dictionary containing tab frames.
        config (Config): Configuration manager for persistent parameters.
        processor (Processor): Processor instance for operations.
        visualizer (Visualizer): Visualizer instance for visualization.
    """
    build_tab1(tabs["tab1"], config, visualizer,dummy_processor)
    build_tab2(tabs["tab2"], config, processor)
    build_tab3(tabs["tab3"], config, processor)
    build_tab4(tabs["tab4"], config, processor)

def build_tab1(tab, config, visualizer,dummy_processor):
    """Builds the Tracking Visualization tab."""
    ttk.Label(tab, text="Tracking Visualization").grid(column=0, row=0, padx=10, pady=10)
    pcap_file = StringVar(value=config.get_param("tab1")["pcap_file"])
    start_date = StringVar(value="0")
    ttk.Entry(tab, textvariable=pcap_file, width=50).grid(column=1, row=0)
    ttk.Button(
        tab,
        text="Select PCAP File",
        command=lambda: select_file("tab1", "pcap_file", pcap_file, config, [("PCAP files", "*.pcap")])
    ).grid(column=2, row=0)

    # Buttons
    ttk.Button(
        tab,
        text="Start Visualization",
        command=lambda: dummy_processor.run_tasks(num_tasks=10, multiplier=2)

    ).grid(column=0, row=1, padx=10, pady=10)

    ttk.Button(
        tab,
        text="Terminate",
        command = dummy_processor.terminate_tasks
    ).grid(column=1, row=1, padx=10, pady=10)

    ttk.Button(
            tab,
            text="Exam LiDAR Date",
            command = lambda: start_date.set(f'Start Date (utc):{ExamPcapStartTime.get_pcap_start_time(pcap_file.get(),unix_time = False)}')
        ).grid(column=0, row=2, padx=10, pady=10)
    
    ttk.Label(tab, 
              textvariable = start_date
              ).grid(column=1, row=2, padx=0, pady=0)


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
        command=lambda: select_folder("tab2", "output_folder", output_folder, config)
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
        text="Enable Additional Processing",
        variable=if_save_forepoints,
        command= lambda: update_flag(if_save_forepoints),
        onvalue=True,
        offvalue=False
        
        
    ).grid(column=0, row=7, padx=10, pady=10)

    # Buttons
    ttk.Button(
        tab,
        text="Start Batch Processing",
        command=lambda: processor.start_batch_processing(
            batch_folder.get(), output_folder.get(), ref_xyz_file.get(), ref_llh_file.get(), diff_to_utc.get()
        )
    ).grid(column=0, row=6, padx=10, pady=10)


def build_tab3(tab, config, processor):
    """Builds the Geometry Referencing tab."""
    ttk.Label(tab, text="Geometry Referencing").grid(column=0, row=0, padx=10, pady=10)

    traj_folder = StringVar(value=config.get_param("tab3")["traj_folder"])
    output_folder = StringVar(value=config.get_param("tab3")["output_folder"])
    ref_xyz_file = StringVar(value=config.get_param("tab3")["ref_xyz_file_path"])
    ref_llh_file = StringVar(value=config.get_param("tab3")["ref_llh_file_path"])

    ttk.Label(tab, text="Trajectory Folder").grid(column=0, row=1)
    ttk.Entry(tab, textvariable=traj_folder, width=50).grid(column=1, row=1)
    ttk.Button(
        tab,
        text="Select Folder",
        command=lambda: select_folder("tab3", "traj_folder", traj_folder, config)
    ).grid(column=2, row=1)

    ttk.Label(tab, text="Output Folder").grid(column=0, row=2)
    ttk.Entry(tab, textvariable=output_folder, width=50).grid(column=1, row=2)
    ttk.Button(
        tab,
        text="Select Folder",
        command=lambda: select_folder("tab3", "output_folder", output_folder, config)
    ).grid(column=2, row=2)

    ttk.Label(tab, text="Reference XYZ File").grid(column=0, row=3)
    ttk.Entry(tab, textvariable=ref_xyz_file, width=50).grid(column=1, row=3)
    ttk.Button(
        tab,
        text="Select File",
        command=lambda: select_file("tab3", "ref_xyz_file_path", ref_xyz_file, config, [("CSV files", "*.csv")])
    ).grid(column=2, row=3)

    ttk.Label(tab, text="Reference LLH File").grid(column=0, row=4)
    ttk.Entry(tab, textvariable=ref_llh_file, width=50).grid(column=1, row=4)
    ttk.Button(
        tab,
        text="Select File",
        command=lambda: select_file("tab3", "ref_llh_file_path", ref_llh_file, config, [("CSV files", "*.csv")])
    ).grid(column=2, row=4)

    ttk.Button(
        tab,
        text="Start Geometry Referencing",
        command=lambda: processor.start_geometry_referencing(
            traj_folder.get(), output_folder.get(), ref_xyz_file.get(), ref_llh_file.get()
        )
    ).grid(column=0, row=5, padx=10, pady=10)


def build_tab4(tab, config, processor):
    """Builds the PCAP Clipping tab."""
    ttk.Label(tab, text="PCAP Clipping").grid(column=0, row=0, padx=10, pady=10)

    pcap_folder = StringVar(value=config.get_param("tab4")["pcap_folder"])
    output_folder = StringVar(value=config.get_param("tab4")["output_folder"])
    time_reference_file = StringVar(value=config.get_param("tab4")["time_reference_file_path"])

    ttk.Label(tab, text="PCAP Folder").grid(column=0, row=1)
    ttk.Entry(tab, textvariable=pcap_folder, width=50).grid(column=1, row=1)
    ttk.Button(
        tab,
        text="Select Folder",
        command=lambda: select_folder("tab4", "pcap_folder", pcap_folder, config)
    ).grid(column=2, row=1)

    ttk.Label(tab, text="Output Folder").grid(column=0, row=2)
    ttk.Entry(tab, textvariable=output_folder, width=50).grid(column=1, row=2)
    ttk.Button(
        tab,
        text="Select Folder",
        command=lambda: select_folder("tab4", "output_folder", output_folder, config)
    ).grid(column=2, row=2)

    ttk.Label(tab, text="Time Reference File").grid(column=0, row=3)
    ttk.Entry(tab, textvariable=time_reference_file, width=50).grid(column=1, row=3)
    ttk.Button(
        tab,
        text="Select File",
        command=lambda: select_file("tab4", "time_reference_file", time_reference_file, config, [("CSV files", "*.csv")])
    ).grid(column=2, row=3)

    ttk.Button(
        tab,
        text="Start PCAP Clipping",
        command=lambda: processor.start_pcap_clipping(pcap_folder.get(), output_folder.get(), time_reference_file.get())
    ).grid(column=0, row=4, padx=10, pady=10)


def select_file(tab, param, string_var, config, file_types):
    """Handles file selection."""
    file_path = filedialog.askopenfilename(filetypes=file_types)
    if file_path:
        string_var.set(file_path)
        config.set_param(tab, param, file_path)
        # showinfo("File Selected", f"Selected file: {file_path}")


def select_folder(tab, param, string_var, config):
    """Handles folder selection."""
    folder_path = filedialog.askdirectory()
    if folder_path:
        string_var.set(folder_path)
        config.set_param(tab, param, folder_path)
        # showinfo("Folder Selected", f"Selected folder: {folder_path}")
