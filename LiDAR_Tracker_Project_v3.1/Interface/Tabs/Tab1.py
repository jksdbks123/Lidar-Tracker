
from Utils import ExamPcapStartTime
from tkinter import ttk
from tkinter import StringVar, IntVar, BooleanVar,DoubleVar
from File_manager import update_flag, select_file, select_folder

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