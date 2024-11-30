import os
import sys
from tkinter import ttk
from tkinter import StringVar, IntVar, BooleanVar,DoubleVar
from tkinter import filedialog
from tkinter.messagebox import showinfo
from Tabs.Tab1 import build_tab1
from Tabs.Tab2 import build_tab2
from Tabs.Tab3 import build_tab3
from Tabs.Tab4 import build_tab4

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
