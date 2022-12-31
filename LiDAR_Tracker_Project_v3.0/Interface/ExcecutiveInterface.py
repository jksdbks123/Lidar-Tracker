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
from multiprocessing import Queue, Value
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
        self.StartFrameIndEntry_Tab1 = None
        self.EndFrameIndEntry_Tab1 = None
        self.StartFrameInd_Tab1 = tk.IntVar()
        self.EndFrameInd_Tab1 = tk.IntVar()
        self.win_size_x,self.win_size_y = tk.IntVar(),tk.IntVar()
        self.win_size_xEntry,self.win_size_yEntry = None,None
        self.eps = tk.DoubleVar()
        self.epsEntry = None
        self.min_samples =  tk.IntVar()
        self.min_samplesEntry = None
        self.bck_update_frame =  tk.IntVar()
        self.bck_update_frameEntry = None
        self.N =  tk.IntVar()
        self.NEntry = None 
        self.d_thred =  tk.DoubleVar()
        self.d_thredEntry = None
        self.bck_n =  tk.IntVar()
        self.bck_nEntry = None
        self.bck_radius =  tk.DoubleVar()
        self.bck_radiusEntry = None
        self.missing_thred =  tk.IntVar()
        self.missing_thredEntry = None
        self.LoadPcapThread = None
        self.ClearEvent_Tab1 = None
        self.StopVisEvent_Tab1 = None
        self.isAllFrameLoaded = False
        self.LoadedFrame = 0
        self.FrameStatus = tk.StringVar()
        self.FrameStatus.set("Loaded Frames:{LoadedFrame}".format(LoadedFrame = self.LoadedFrame))
        self.FrameStatusLabel =  ttk.Label(self.tab1, textvariable= self.FrameStatus)
        self.FrameStatusLabel.grid(column=0, row=7, padx=0, pady=0)
        # Visulization Tab 1 
        ttk.Label(self.tab2, text="Generate Trajectories From Single Pcap File").grid(column=0, row=0, padx=30, pady=30)
        # Visulization Tab 1 
        ttk.Label(self.tab3, text="Batch Trajectories Generation").grid(column=0, row=0, padx=30, pady=30)
        self.tabControl.pack(expand = 1, fill ="both")
        self.CreateTab1()
        #Memory
        self.Off_tracking_pool = {}
        self.aggregated_maps = [] 



    def CreateTab1(self):
        """
        Path Input
        """
        ttk.Label(self.tab1, text= "Pcap Path").grid(column=1, row=0, padx=0, pady=0)
        self.PcapPathEntry_Tab1 = ttk.Entry(self.tab1,text = "Select Pcap File")
        self.PcapPathEntry_Tab1.grid(column=1, row=1, padx=0, pady=0)
        selectInputButton = ttk.Button(
            self.tab1,
            text='Select Pcap File',
            command = self.selectPcap
        )
        selectInputButton.grid(column=0, row=1, padx=0, pady=0)
        """
        Frame Input
        """
        ttk.Label(self.tab1, text= "Start Index").grid(column=2, row=0, padx=0, pady=0)
        ttk.Label(self.tab1, text= "End Index").grid(column=3, row=0, padx=0, pady=0)
        self.StartFrameIndEntry_Tab1 = ttk.Entry(self.tab1,textvariable = self.StartFrameInd_Tab1,width=5)
        self.StartFrameIndEntry_Tab1.grid(column=2, row=1, padx=0, pady=0)
        self.StartFrameInd_Tab1.set(0)
        self.EndFrameIndEntry_Tab1 = ttk.Entry(self.tab1,textvariable = self.EndFrameInd_Tab1,width=5)
        self.EndFrameIndEntry_Tab1.grid(column=3, row=1, padx=0, pady=0)
        self.EndFrameInd_Tab1.set(0)
        """
        Tracking Parameters
        """
        ttk.Label(self.tab1, text= "WinSizeX").grid(column=1, row=2, padx=0, pady=0)
        ttk.Label(self.tab1, text= "WinSizeY").grid(column=2, row=2, padx=0, pady=0)
        self.win_size_xEntry = ttk.Entry(self.tab1,textvariable = self.win_size_x,width=5)
        self.win_size_xEntry.grid(column=1, row=3, padx=0, pady=0)
        self.win_size_x.set(7)
        self.win_size_yEntry = ttk.Entry(self.tab1,textvariable = self.win_size_y,width=5)
        self.win_size_yEntry.grid(column=2, row=3, padx=0, pady=0)
        self.win_size_y.set(13)
        ttk.Label(self.tab1, text= "eps").grid(column=3, row=2, padx=0, pady=0)
        self.epsEntry = ttk.Entry(self.tab1,textvariable = self.eps,width=5)
        self.epsEntry.grid(column=3, row=3, padx=0, pady=0)
        self.eps.set(1.5)
        ttk.Label(self.tab1, text= "MinSample").grid(column=1, row=4, padx=0, pady=0)
        self.min_samplesEntry = ttk.Entry(self.tab1,textvariable = self.min_samples,width=5)
        self.min_samplesEntry.grid(column=1, row=5, padx=0, pady=0)
        self.min_samples.set(5)
        ttk.Label(self.tab1, text= "BckUpd#").grid(column=2, row=4, padx=0, pady=0)
        self.bck_update_frameEntry = ttk.Entry(self.tab1,textvariable = self.bck_update_frame,width=5)
        self.bck_update_frameEntry.grid(column=2, row=5, padx=0, pady=0)
        self.bck_update_frame.set(500)
        ttk.Label(self.tab1, text= "BckSpl#").grid(column=3, row=4, padx=0, pady=0)
        self.NEntry = ttk.Entry(self.tab1,textvariable = self.N,width=5)
        self.NEntry.grid(column=3, row=5, padx=0, pady=0)
        self.N.set(20)
        ttk.Label(self.tab1, text= "Ts").grid(column=1, row=6, padx=0, pady=0)
        self.d_thredEntry = ttk.Entry(self.tab1,textvariable = self.d_thred,width=5)
        self.d_thredEntry.grid(column=1, row=7, padx=0, pady=0)
        self.d_thred.set(0.056)
        ttk.Label(self.tab1, text= "BckVir#").grid(column=2, row=6, padx=0, pady=0)
        self.bck_nEntry = ttk.Entry(self.tab1,textvariable = self.bck_n,width=5)
        self.bck_nEntry.grid(column=2, row=7, padx=0, pady=0)
        self.bck_n.set(3)
        ttk.Label(self.tab1, text= "BckRad").grid(column=3, row=6, padx=0, pady=0)
        self.bck_radiusEntry = ttk.Entry(self.tab1,textvariable = self.bck_radius,width=5)
        self.bck_radiusEntry.grid(column=3, row=7, padx=0, pady=0)
        self.bck_radius.set(0.9)
        ttk.Label(self.tab1, text= "Connectivity").grid(column=4, row=6, padx=0, pady=0)
        self.missing_thredEntry = ttk.Entry(self.tab1,textvariable = self.missing_thred,width=5)
        self.missing_thredEntry.grid(column=4, row=7, padx=0, pady=0)
        self.missing_thred.set(10)
        """
        Buttoms
        """
        LoadButton = ttk.Button(
            self.tab1,
            text='LoadPcap',
            command=self.LoadPcap
        )
        LoadButton.grid(column=0, row=3, padx=0, pady=0)

        VisButton = ttk.Button(
            self.tab1,
            text='Preview',
            command=self.CreateAni
        )
        VisButton.grid(column=0, row=4, padx=0, pady=0)

        StopVisButtom = ttk.Button(
            self.tab1,
            text='Stop',
            command=self.TerminateTab1
        )
        StopVisButtom.grid(column=0, row=5, padx=0, pady=0)  

        TerminateButton = ttk.Button(
            self.tab1,
            text='Clear Temp',
            command=self.TerminateTab1
        )
        TerminateButton.grid(column=0, row=6, padx=0, pady=0)    
        """
        Buttoms
        """

    def LoadPcap(self):
        self.ClearEvent_Tab1 = Event()
        self.LoadPcapThread = Thread(target=self.Loading,args=(self.ClearEvent_Tab1,))
        self.LoadPcapThread.start()
        
    def TerminateTab1(self): # kill all thread
        self.ClearEvent_Tab1.set()
        self.FrameStatus.set("Loaded Frames:0")
    

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
                Td_map,Int_map = Frame
                self.LoadedFrame += 1
                self.FrameStatus.set("Loaded Frames:{}".format(len(self.aggregated_maps)))
                self.aggregated_maps.append(Td_map)
        else:
            print('Path not exists')
    def Tracking(self):
        pass
    def CreateAni(self):
        vis = op3.visualization.Visualizer()
        vis.create_window()
        StartFrame = self.StartFrameInd_Tab1.get()
        EndFrame = self.EndFrameInd_Tab1.get()
        if StartFrame < 0:
            StartFrame = 0 
        if EndFrame > (len(self.aggregated_maps) - 1):
            EndFrame = len(self.aggregated_maps) - 1
        source = get_pcd_colored(self.aggregated_maps[StartFrame],np.ones((32,1800)))
        vis.add_geometry(source)

        # while not event.is_set():
        for i in range(StartFrame+1,EndFrame):
            print(i)
            # if event.is_set():
            #     break
            pcd = get_pcd_colored(self.aggregated_maps[i],np.ones((32,1800)))
            source.points = pcd.points
            source.colors = pcd.colors
            vis.update_geometry(source)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)

        vis.destroy_window()
            
        
        
class Visulizer():
    def __init__(self,Q):
        self.vis = op3.visualization.Visualizer()
        self.FrameQueue = Q
        # self.vis.create_window()
        # pcd = op3.io.read_point_cloud('./000006.pcd')
        # pcd.points = op3.utility.Vector3dVector(np.zeros((57600,3)))
        # pcd.colors = op3.utility.Vector3dVector(255*np.ones((57600,3)))
        # self.source = pcd
        # self.vis.add_geometry(self.source)
    def run(self):
        self.vis.create_window()
        # Td_map = self.aggregated_maps[self.InputFrameInd_Tab1.get()]
        # pcd = get_pcd_colored(Td_map,np.ones((32,1800)))
        # self.source.points = pcd.points
        # self.source.colors = pcd.colors
        # print(np.asarray(self.source.points))
        while True:
            # self.vis.update_geometry(self.source)
            # self.vis.poll_events()
            # self.vis.update_renderer()
            # print(np.asarray(self.source.points))
            print(self.FrameQueue.qsize())




        


if __name__ == "__main__":
    
    GUI = Interface()
    GUI.root.mainloop()



    