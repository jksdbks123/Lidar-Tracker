import json
import os

"""
Tab 1: Tracking Visualization. may connect to our visualizer. parameters: pcap_file (selcted by filedialog)
Tab 2: Batch Processing. parameters: batch_folder (selected by filedialog), output_folder (selected by filedialog), 
ref_xyz_file_path (selected by filedialog), ref_llh_file_path (selected by filedialog), Diff2UTC (default -8, entry), potentially have more to be added
Tab 3: Geometry Referencing.parameters: traj_folder (selected by filedialog), output_folder (selected by filedialog),ref_xyz_file_path (selected by filedialog), ref_llh_file_path (selected by filedialog)
Tab 4: PCAP Clipping, clip pcap files. parameters: pcap_folder (selected by filedialog), output_folder (selected by filedialog), time_reference_file (selected by filedialog)
"""

class Config:
    CONFIG_FILE = "config.json"
    def __init__(self):
        self.params = { # Default configuration
            # filer parameters in four tabs
            "tab1": {'pcap_file': ""},
            "tab2": {'batch_folder': "", 'output_folder': "", 'ref_xyz_file_path': "", 'ref_llh_file_path': "", 'Diff2UTC': -8},
            "tab3": {'traj_folder': "", 'output_folder': "", 'ref_xyz_file_path': "", 'ref_llh_file_path': ""},
            "tab4": {'pcap_folder': "", 'output_folder': "", 'time_reference_file_path': ""},
        }
        self.load_config()
        if not os.path.exists(self.CONFIG_FILE):
            # crrate a new config file 
            self.save_config()
    def load_config(self):
        if os.path.exists(self.CONFIG_FILE):
            with open(self.CONFIG_FILE, "r") as file:
                self.params = json.load(file)
        # go through all elements, if it ends with file or folder and is empty, set it to empty string
        for tab in self.params:
            for param in self.params[tab]:
                if param.endswith("_file") or param.endswith("_folder"):
                    # test if the file or folder exists
                    if not os.path.exists(self.params[tab][param]):
                        self.params[tab][param] = ""
                        
    def save_config(self):
        with open(self.CONFIG_FILE, "w") as file:
            json.dump(self.params, file, indent=4)

    def set_param(self, tab, param, value):
        self.params[tab][param] = value
        self.save_config()

    def get_param(self, key):
        return self.params.get(key, "")
