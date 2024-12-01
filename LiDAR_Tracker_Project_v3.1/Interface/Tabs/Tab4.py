from tkinter import ttk
from tkinter import StringVar, IntVar, BooleanVar,DoubleVar
from File_manager import update_flag, select_file, select_folder

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