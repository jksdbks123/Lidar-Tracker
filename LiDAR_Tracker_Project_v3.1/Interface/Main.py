import sys,os
# consider we are in the Interface folder, root folder is LiDAR_Tracker_Project_v3.1
functions_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'Tabs'))
utils_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'Utils'))
if functions_folder_path not in sys.path:
    sys.path.append(functions_folder_path)
if utils_folder_path not in sys.path:
    sys.path.append(utils_folder_path)

from tkinter import ttk
import tkinter as tk
from Interface_builder import build_interface
from Utils.config import Config
from Utils.Processing import Processor
from Utils.Visualizer import Visualizer
from Utils.Tracker import MOT
from Utils.dummy_processing import DummyProcessor
class Interface:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Tracking Software')
        self.root.geometry('1000x500')

        # Persistent configuration
        self.config = Config()

        # Tabs
        self.tabControl = ttk.Notebook(self.root)
        self.tabs = {
            "tab1": ttk.Frame(self.tabControl),
            "tab2": ttk.Frame(self.tabControl),
            "tab3": ttk.Frame(self.tabControl),
            "tab4": ttk.Frame(self.tabControl),
        }

        self.processor = Processor()
        self.visualizer = Visualizer()
        # self.tracker = MOT()
        self.dummy_processor = DummyProcessor()

        tab_name_list = [ "Visualization", "Batch Processing", "Geometry Referencing", "PCAP Clipping"]
        for tab_name in tab_name_list:
            self.tabControl.add(self.tabs[f"tab{tab_name_list.index(tab_name)+1}"], text=tab_name)

        self.tabControl.pack(expand=1, fill="both")

        # Build the interface
        build_interface(self.tabs, self.config, self.processor, self.visualizer,self.dummy_processor)

if __name__ == "__main__":
    app = Interface()
    app.root.mainloop()
