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
        self.general_params = { # Default configuration
            # filer parameters in four tabs
            "tab1": {'pcap_file': ""},
            "tab2": {'batch_folder': "", 'output_folder': "", 'ref_xyz_file_path': "", 'ref_llh_file_path': "", 'Diff2UTC': -8, 'SaveForepoints': False},
            "tab3": {'traj_folder': "", 'output_folder': "", 'ref_xyz_file_path': "", 'ref_llh_file_path': ""},
            "tab4": {'pcap_folder': "", 'output_folder': "", 'time_reference_file_path': ""},
        }
        
        if not os.path.exists(self.CONFIG_FILE):
            # crrate a new config file 
            self.save_config()
        else:
            self.load_config()

    def load_config(self):
        with open(self.CONFIG_FILE, "r") as file:
            temp_config = json.load(file)
        # go through all elements, if it ends with file or folder and is empty, set it to empty string
        for tab in self.general_params:
            for param in self.general_params[tab]:
                # test if the file or folder exists
                if param not in temp_config[tab]:
                    # make sure fill in the correct data type
                    temp_config[tab][param] = self.general_params[tab][param]
                if param.endswith("_file") or param.endswith("_folder"):
                    if temp_config[tab][param] not in self.general_params[tab][param]:
                        temp_config[tab][param] = ""
                        
        self.general_params = temp_config

                    
       

    def save_config(self):
        with open(self.CONFIG_FILE, "w") as file:
            json.dump(self.general_params, file, indent=4)

    def set_param(self, tab, param, value):
        self.general_params[tab][param] = value
        self.save_config()

    def get_param(self, key):
        return self.general_params.get(key, "")
