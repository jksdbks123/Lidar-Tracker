import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter import filedialog
import pandas as pd
import dpkt
import numpy as np
import os
from threading import *
from multiprocessing import Process

import time
from BfTableGenerator import TDmapLoader
from tqdm import tqdm
import open3d as op3
from VisulizerTools import *
class Interface():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Super Tracker Toolbox -- powered by czh')
        self.root.resizable(True,True)
        self.root.geometry('700x300')
        # Tab1: Visulization; Tab2: Generate trajectories from single pcap file; Tab3: Batch Trajs generation 
        self.tabControl = ttk.Notebook(self.root)
        self.tab1 = ttk.Frame(self.tabControl)
        self.tab2 = ttk.Frame(self.tabControl)
        self.tab3 = ttk.Frame(self.tabControl)

        self.tabControl.add(self.tab1, text='Tracking Visulization')
        self.tabControl.add(self.tab2, text='Single Generation')
        self.tabControl.add(self.tab3, text='Batch Generation')
        # Visulization Tab 1
        ttk.Label(self.tab1, text= "Tracking Visulization").grid(column=0, row=0, padx=30, pady=10)
        self.PcapPathEntry_Tab1 = None
        self.InputFrameInd_Tab1 = tk.IntVar()

        self.InputFrameEntry_Tab1 = None
        self.VisulizationThread = None
        self.LoadPcapThread = None
        self.TerminateEvent_Tab1 = None
        self.isAllFrameLoaded = False
        self.LoadedFrame = 0
        self.CurrentFrame = 0
        self.FrameStatus = tk.StringVar()
        self.FrameStatus.set("{CurrentFrame}/{LoadedFrame}".format(CurrentFrame = self.CurrentFrame,LoadedFrame = self.LoadedFrame))
        self.FrameStatusLabel =  ttk.Label(self.tab1, textvariable= self.FrameStatus)
        self.FrameStatusLabel.grid(column=3, row=4, padx=30, pady=10)
        # Visulization Tab 1 
        ttk.Label(self.tab2, text="Generate Trajectories From Single Pcap File").grid(column=0, row=0, padx=30, pady=30)
        # Visulization Tab 1 
        ttk.Label(self.tab3, text="Batch Trajectories Generation").grid(column=0, row=0, padx=30, pady=30)
        self.tabControl.pack(expand = 1, fill ="both")
        self.CreateTab1()
        #Memory
        self.aggregated_maps = []



    def CreateTab1(self):
        ttk.Label(self.tab1, text= "Pcap Path").grid(column=1, row=0, padx=0, pady=0)
        self.PcapPathEntry_Tab1 = ttk.Entry(self.tab1,text = "Select Pcap File")
        self.PcapPathEntry_Tab1.grid(column=1, row=1, padx=0, pady=0)
        selectInputButton = ttk.Button(
            self.tab1,
            text='Select Pcap File',
            command = self.selectPcap
        )
        selectInputButton.grid(column=0, row=1, padx=0, pady=0)

        ttk.Label(self.tab1, text= "Current Frame Index").grid(column=2, row=0, padx=0, pady=0)
        self.InputFrameEntry_Tab1 = ttk.Entry(self.tab1,textvariable = self.InputFrameInd_Tab1)
        self.InputFrameEntry_Tab1.grid(column=2, row=1, padx=0, pady=0)
        self.InputFrameInd_Tab1.set(0)

        LoadButton = ttk.Button(
            self.tab1,
            text='LoadPcap',
            command=self.LoadPcap
        )
        LoadButton.grid(column=0, row=3, padx=0, pady=0)

        VisButton = ttk.Button(
            self.tab1,
            text='Visulize',
            command=self.OpenVis
        )
        VisButton.grid(column=0, row=4, padx=0, pady=0)

        TerminateButton = ttk.Button(
            self.tab1,
            text='Terminate',
            command=self.TerminateTab1
        )
        TerminateButton.grid(column=0, row=5, padx=0, pady=0)    

        Forward_ = ttk.Button(
            self.tab1,
            text='Forward-',
            command = self.PlayForward_
        )
        Forward_.grid(column=5, row=3, padx=0, pady=0)  

        Forward = ttk.Button(
            self.tab1,
            text='Forward',
            command = self.PlayForward
        )
        Forward.grid(column=4, row=3, padx=0, pady=0)  

        Stop = ttk.Button(
            self.tab1,
            text='Stop',
            command = self.Stop
        )
        Stop.grid(column=3, row=3, padx=0, pady=0)  

        Backward = ttk.Button(
            self.tab1,
            text='Backward',
            command = self.PlayBackward
        )
        Backward.grid(column=2, row=3, padx=0, pady=0)  

        Backward_ = ttk.Button(
            self.tab1,
            text='_Backward',
            command = self.PlayForward_
        )
        Backward_.grid(column=1, row=3, padx=0, pady=0)  

    def LoadPcap(self):
        self.TerminateEvent_Tab1 = Event()
        self.LoadPcapThread = Thread(target=self.Loading,args=(self.TerminateEvent_Tab1,))
        self.LoadPcapThread.start()
        
    def TerminateTab1(self): # kill all thread
        self.TerminateEvent_Tab1.set()
        self.FrameStatus.set("0/0")
        

    def PlayForward(self):
        pass
    def PlayForward_(self):
        pass
    def PlayBackward(self):
        pass
    def PlayBackward_(self):
        pass
    def Stop(self):
        pass

    def selectPcap(self):
        filetypes = (
            ('pcap files', '*.pcap'),
        )

        filename = filedialog.askopenfilenames(
            title='Open a pcap file',
            initialdir='/',
            filetypes=filetypes)

        self.PcapPathEntry_Tab1.delete(0,'end')
        self.PcapPathEntry_Tab1.insert(0,filename)

    def Loading(self,event):
        if os.path.exists(self.PcapPathEntry_Tab1.get()):
            # self.VisulizationThread = Thread(target=self.Visulization,args=(self.TerminateEvent_Tab1,))
            # self.VisulizationThread.start()
            frame_gen = TDmapLoader(self.PcapPathEntry_Tab1.get()).frame_gen()
            while True:
                Frame = next(frame_gen)
                if (Frame is None):
                    break 
                if event.is_set(): 
                    self.CurrentFrame = 0 
                    self.LoadedFrame = 0
                    self.aggregated_maps = []
                    break
                Td_map,Int_map = Frame
                self.LoadedFrame += 1
                self.FrameStatus.set("{}/{}".format(self.CurrentFrame,self.LoadedFrame))
                self.aggregated_maps.append(Td_map)
        else:
            print('Path not exists')
    def OpenVis(self):
        pass
        

class Visulizer():
    def __init__(self):
        self.vis = op3.visualization.Visualizer()
        # self.vis.create_window()

        # pcd = op3.io.read_point_cloud('./000006.pcd')
        # pcd.points = op3.utility.Vector3dVector(np.zeros((57600,3)))
        # pcd.colors = op3.utility.Vector3dVector(255*np.ones((57600,3)))
        # self.source = pcd
        # self.vis.add_geometry(self.source)
    def Visulization(self):
        Td_map = self.aggregated_maps[self.InputFrameInd_Tab1.get()]
        pcd = get_pcd_colored(Td_map,np.ones((32,1800)))
        self.source.points = pcd.points
        self.source.colors = pcd.colors
        print(np.asarray(self.source.points))
        while True:
            self.vis.update_geometry(self.source)
            self.vis.poll_events()
            self.vis.update_renderer()
            # print(np.asarray(self.source.points))




        


if __name__ == "__main__":
    tracker = Tracker()
    tracker.run()



    