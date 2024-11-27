import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter import filedialog
import pandas as pd
import dpkt
import numpy as np
import os
from threading import Thread,Event
from multiprocessing import Process
from multiprocessing import Queue, Value
import time
from BfTableGenerator import TDmapLoader
import MOTLite 
from MOT_TD_BCKONLIONE import MOT
from tqdm import tqdm
import open3d as op3
from VisulizerTools import *
from p_tqdm import p_umap
from functools import partial
import pandas as pd
def get_pcap_time(input_file_path,time_label):
    with open(input_file_path, 'rb') as fpcap:
        try:
            lidar_reader = dpkt.pcap.Reader(fpcap)
            while True:
                try:
                    ts,buf = next(lidar_reader)
                    eth = dpkt.ethernet.Ethernet(buf)
                except:
                    break
                if eth.type == 2048: # for ipv4
                    if type(eth.data.data) == dpkt.udp.UDP:
                        data = eth.data.data.data
                        packet_status = eth.data.data.sport
                        if packet_status == 2368:
                            if len(data) == 1206:
                                start_timestamp = ts
                                break
        except dpkt.dpkt.NeedData:
            print('Invalid')
    
    time_label.set(str(pd.to_datetime(start_timestamp,unit='s')))

def run_clipping(pcap_path,target_frame,pcap_name, output_path):
    # load packets from pcap until the last frame in the end_frames
    # target_frame: a 2 x 2 np.array, with first colume start frame and second colume end frame
    print('o:',output_path)
    # outfolder = os.path.join(output_path,pcap_path.split('.')[0])
    folder_name = pcap_name.split('.')[0]
    outfolder = os.path.join(output_path,folder_name)
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
    packets = []
    tses = []
    frame_index = []
    cur_ind = 0
    print('Processing', outfolder)
    with open(pcap_path, 'rb') as fpcap:
        lidar_reader = dpkt.pcap.Reader(fpcap)
        try:
            ts,buf = next(lidar_reader)
            eth = dpkt.ethernet.Ethernet(buf)
            next_ts = ts + 0.1
            packets.append(eth)
            tses.append(ts)
            frame_index.append(cur_ind)
        except:
            pass
        while True:
            if cur_ind > target_frame[:,1].max():
                break
            try:
                frame_index.append(cur_ind)
                ts,buf = next(lidar_reader)
                eth = dpkt.ethernet.Ethernet(buf)
                packets.append(eth)
                tses.append(ts)
                if ts > next_ts:
                    cur_ind += 1
                    next_ts += 0.1
            except:
                break
    # save the snippets to specified folder
    frame_index = np.array(frame_index)
    for i in range(len(target_frame)):
        with open(os.path.join(outfolder,'{}_{}.pcap'.format(target_frame[i,0],target_frame[i,1])),'wb') as wpcap:
            lidar_writer = dpkt.pcap.Writer(wpcap)
            start_ind = np.where(frame_index == target_frame[i,0])[0][0]
            end_ind = np.where(frame_index == target_frame[i,1])[0][0]
            for f_ind in range(start_ind,end_ind):
                lidar_writer.writepkt(packets[f_ind],ts = tses[f_ind])

def run_mot(mot,ref_LLH,ref_xyz,utc_diff):
    mot.initialization()
    if mot.thred_map is not None:
        mot.mot_tracking(mot.frame_gen)
        if mot.if_pcap_valid:
            save_result(mot.Off_tracking_pool,ref_LLH,ref_xyz,mot.traj_path,mot.start_timestamp, utc_diff)
            print(mot.traj_path)

def run_georef(traj_path,output_path,T):
    traj = pd.read_csv(traj_path)
    temp_xyz = np.array(traj.loc[:,['Coord_X','Coord_Y','Coord_Z']])
    normal_ind = []
    for i,e in enumerate(temp_xyz):
        try:
            temp_xyz[i] = e.astype(np.float64)
            normal_ind.append(i)
        except:
            pass 
    temp_xyz = temp_xyz[normal_ind].astype(np.float64)
    traj = traj.iloc[normal_ind].reset_index(drop = True)
    LLH = convert_LLH(temp_xyz,T)
    traj.loc[:,['Longitude','Latitude','Elevation']] = LLH
    traj.to_csv(output_path,index=False)

class Interface():
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('A Bad Bad Tracker --- powered by Zhihui')
        self.root.resizable(True,True)
        self.root.geometry('1000x500')
        # Tab1: Visulization; Tab2: Generate trajectories from single pcap file; Tab3: Batch Trajs generation 
        self.tabControl = ttk.Notebook(self.root)
        self.tab1 = ttk.Frame(self.tabControl)
        self.tab2 = ttk.Frame(self.tabControl)
        self.tab3 = ttk.Frame(self.tabControl)
        self.tab4 = ttk.Frame(self.tabControl)

        self.tabControl.add(self.tab1, text='Tracking Visulization')
        self.tabControl.add(self.tab2, text='Batch Generation')
        self.tabControl.add(self.tab3, text='Geometry Referencing')
        self.tabControl.add(self.tab4, text='Pcap Clipping')
        
        # Visulization Tab 1
        ttk.Label(self.tab1, text= "Tracking Visulization").grid(column=0, row=0, padx=30, pady=10)
        self.PcapPathEntry_Tab1 = None
        self.StartFrameIndEntry_Tab1 = None
        self.EndFrameIndEntry_Tab1 = None
        self.StartFrameInd_Tab1 = tk.IntVar()
        self.EndFrameInd_Tab1 = tk.IntVar()
        self.PcapTime_Tab1 = tk.StringVar()
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
        self.TrackingThread = None
        self.BatchTrackingThread = None
        self.GeoRefThread = None
        self.PcapClippingThread = None
        self.ClearEvent_Tab1 = None
        self.TrackingTerminateEvent_Tab1 = None
        self.StopVisEvent_Tab1 = None
        self.isAllFrameLoaded = False
        self.LoadedFrame = 0
        self.TrackedFrame = 0
        self.FrameStatus = tk.StringVar() # loaded frame status
        self.FrameStatus.set("Loaded Frames:{LoadedFrame}".format(LoadedFrame = self.LoadedFrame))
        self.FrameStatusLabel =  ttk.Label(self.tab1, textvariable= self.FrameStatus)
        self.FrameStatusLabel.grid(column=0, row=9, padx=0, pady=0)
        self.TrackingStatus = tk.StringVar() # loaded frame status
        self.TrackingStatus.set("Tracked Frames:{TrackedFrame}".format(TrackedFrame = self.TrackedFrame))
        self.TrackingStatusLabel =  ttk.Label(self.tab1, textvariable= self.TrackingStatus)
        self.TrackingStatusLabel.grid(column=0, row=8, padx=0, pady=0)
        
        # Visulization Tab 3 
        self.tabControl.pack(expand = 1, fill ="both")
        #Memory
        self.mot = None
        self.aggregated_maps = [] 
        self.start_timestamp = 0
        self.if_pcap_valid = True
        self.CreateTab1()
        """
        Tab2
        """
        self.cpu_nEntryTab2 = None
        self.cpu_nTab2 = tk.IntVar()
        self.UTC_diffEntry = None
        self.UTC_diff = tk.IntVar()
        self.PcapPathEntry_Tab2 = None
        self.RefXyzEntry_Tab2 = None
        self.RefLlhEntry_Tab2 = None
        self.OutputEntry_Tab2 = None
        self.CreateTab2()
        
        """
        Tab3
        """
        self.cpu_nEntryTab3 = None
        self.cpu_nTab3 = tk.IntVar()
        self.TrajPathEntry_Tab3 = None
        self.RefXyzEntry_Tab3 = None
        self.RefLlhEntry_Tab3 = None
        self.OutputEntry_Tab3 = None
        self.CreateTab3()
        """
        Tab4
        """
        self.cpu_nEntryTab4 = None
        self.cpu_nTab4 = tk.IntVar()
        self.PcapPathEntry_Tab4 = None
        self.TimeStampKey_Tab4 = tk.StringVar()
        self.TimeStampKeyEntry_Tab4 = None
        self.FrameKey_Tab4 = tk.StringVar()
        self.FrameKeyEntry_Tab4 = None
        self.OutputEntry_Tab4 = None
        self.TimeRefFile = tk.StringVar()
        self.TimeRefFileEntry_Tab4 = None
        self.TimeInterval_Tab4 = tk.IntVar() #sec
        self.TimeIntervalEntry_Tab4 = None
        
        self.CreateTab4()

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
        ttk.Label(self.tab1, text='Pcap Time').grid(column=4, row=0, padx=0, pady=0)
        ttk.Label(self.tab1, textvariable= self.PcapTime_Tab1).grid(column=4, row=1, padx=0, pady=0)
        self.PcapTime_Tab1.set('0')
        """
        Buttoms
        """
        LoadButton = ttk.Button(
            self.tab1,
            text='LoadPcap',
            command=self.LoadPcap
        )
        LoadButton.grid(column=0, row=3, padx=0, pady=0)
        TrackButton = ttk.Button(
            self.tab1,
            text='Track',
            command=self.Tracking
        )
        TrackButton.grid(column=0, row=4, padx=0, pady=0)
        VisButton = ttk.Button(
            self.tab1,
            text='Preview',
            command=self.CreateAni
        )
        VisButton.grid(column=0, row=5, padx=0, pady=0)

        ClearTempButton = ttk.Button(
            self.tab1,
            text='Clear Temp',
            command=self.ClearUp # clear up temps
        )
        ClearTempButton.grid(column=0, row=6, padx=0, pady=0)
        TrackTerminateButton = ttk.Button(
        self.tab1,
        text='Terminate Tracking',
        command=self.TrackerTerminate
        )
        TrackTerminateButton.grid(column=0, row=7, padx=0, pady=0)


        """
        Buttoms
        """
    def CreateTab2(self):
        ttk.Label(self.tab2, text= "Input Pcap Folder").grid(column=1, row=0, padx=0, pady=0)
        ttk.Label(self.tab2, text= "Output Folder").grid(column=4, row=0, padx=0, pady=0)
        self.PcapPathEntry_Tab2 = ttk.Entry(self.tab2,text = "Select Pcap Folder")
        self.PcapPathEntry_Tab2.grid(column=1, row=1, padx=0, pady=0)
        self.OutputEntry_Tab2 = ttk.Entry(self.tab2,text = "Select Output Folder")
        self.OutputEntry_Tab2.grid(column=4, row=1, padx=0, pady=0)
        self.RefXyzEntry_Tab2 = ttk.Entry(self.tab2,text = "Select xyz ref")
        self.RefXyzEntry_Tab2.grid(column=1, row=2, padx=0, pady=0)
        self.RefLlhEntry_Tab2 = ttk.Entry(self.tab2,text = "Select llh ref")
        self.RefLlhEntry_Tab2.grid(column=1, row=3, padx=0, pady=0)
        selectInputButton = ttk.Button(
            self.tab2,
            text='Select Pcap Folder',
            command = self.selectInputDirectory
        )
        selectInputButton.grid(column=0, row=1, padx=0, pady=0)
        ttk.Label(self.tab2, text= "CPUs").grid(column=2, row=0, padx=0, pady=0)
        self.cpu_nEntryTab2 = ttk.Entry(self.tab2,textvariable = self.cpu_nTab2,width=5)
        self.cpu_nEntryTab2.grid(column=2, row=1, padx=0, pady=0)
        self.cpu_nTab2.set(1)
        ttk.Label(self.tab2, text= "UTC Diff").grid(column=2, row=2, padx=0, pady=0)
        self.UTC_diffEntry = ttk.Entry(self.tab2,textvariable = self.UTC_diff,width=5)
        self.UTC_diffEntry.grid(column=2, row=3, padx=0, pady=0)
        self.UTC_diff.set(-8)

        Select_xyzRef = ttk.Button(
        self.tab2,
        text='SelectRefXyz',
        command=self.selectRefXyzTab2
        )
        Select_xyzRef.grid(column=0, row=2, padx=0, pady=0) 

        Select_llhRef = ttk.Button(
        self.tab2,
        text='SelectRefLlh',
        command=self.selectRefLlhTab2
        )
        Select_llhRef.grid(column=0, row=3, padx=0, pady=0) 
        SelectOutputPath = ttk.Button(
        self.tab2,
        text='Select Output Path',
        command=self.selectOutputDirectoryTab2
        )
        SelectOutputPath.grid(column=3, row=1, padx=0, pady=0) 
        BatchProcess = ttk.Button(
        self.tab2,
        text='Batch Process',
        command=self.StartBatch
        )
        BatchProcess.grid(column=0, row=4, padx=0, pady=0)   

    def CreateTab3(self):
        ttk.Label(self.tab3, text= "Input Trajectories Folder").grid(column=1, row=0, padx=0, pady=0)
        ttk.Label(self.tab3, text= "Output Folder").grid(column=4, row=0, padx=0, pady=0)
        self.TrajPathEntry_Tab3 = ttk.Entry(self.tab3,text = "Select Pcap Folder")
        self.TrajPathEntry_Tab3.grid(column=1, row=1, padx=0, pady=0)
        self.OutputEntry_Tab3 = ttk.Entry(self.tab3,text = "Select Output Folder")
        self.OutputEntry_Tab3.grid(column=4, row=1, padx=0, pady=0)
        self.RefXyzEntry_Tab3 = ttk.Entry(self.tab3,text = "Select xyz ref")
        self.RefXyzEntry_Tab3.grid(column=1, row=2, padx=0, pady=0)
        self.RefLlhEntry_Tab3 = ttk.Entry(self.tab3,text = "Select llh ref")
        self.RefLlhEntry_Tab3.grid(column=1, row=3, padx=0, pady=0)
        ttk.Label(self.tab3, text= "CPUs").grid(column=2, row=0, padx=0, pady=0)
        self.cpu_nEntryTab3 = ttk.Entry(self.tab3,textvariable = self.cpu_nTab3,width=5)
        self.cpu_nEntryTab3.grid(column=2, row=1, padx=0, pady=0)
        self.cpu_nTab3.set(1)
        selectTrajFolderButton = ttk.Button(
            self.tab3,
            text='Select Folder',
            command = self.selectTrajDirectory
        )
        selectTrajFolderButton.grid(column=0, row=1, padx=0, pady=0)
        Select_xyzRef = ttk.Button(
        self.tab3,
        text='SelectRefXyz',
        command=self.selectRefXyzTab3
        )
        Select_xyzRef.grid(column=0, row=2, padx=0, pady=0) 

        Select_llhRef = ttk.Button(
        self.tab3,
        text='SelectRefLlh',
        command=self.selectRefLlhTab3
        )
        Select_llhRef.grid(column=0, row=3, padx=0, pady=0) 
        SelectOutputPath = ttk.Button(
        self.tab3,
        text='Select Output Path',
        command=self.selectOutputDirectoryTab3
        )
        SelectOutputPath.grid(column=3, row=1, padx=0, pady=0) 
        BatchProcess = ttk.Button(
        self.tab3,
        text='Batch Process',
        command=self.StartGeoRef
        )
        BatchProcess.grid(column=0, row=4, padx=0, pady=0)
    def CreateTab4(self):
        ttk.Label(self.tab4, text= "Input Pcap Folder").grid(column=1, row=0, padx=0, pady=0)
        ttk.Label(self.tab4, text= "Output Folder").grid(column=5, row=0, padx=0, pady=0)
        self.PcapPathEntry_Tab4 = ttk.Entry(self.tab4,text = "Select Pcap Folder")
        self.PcapPathEntry_Tab4.grid(column=1, row=1, padx=0, pady=0)
        self.OutputEntry_Tab4 = ttk.Entry(self.tab4,text = "Select Output Folder")
        self.OutputEntry_Tab4.grid(column=5, row=1, padx=0, pady=0)
        ttk.Label(self.tab4, text= "CPUs").grid(column=2, row=0, padx=0, pady=0)
        self.cpu_nEntryTab4 = ttk.Entry(self.tab4,textvariable = self.cpu_nTab4,width=5)
        self.cpu_nEntryTab4.grid(column=2, row=1, padx=0, pady=0)
        self.cpu_nTab4.set(1)
        selectPcapFolderButton = ttk.Button(
            self.tab4,
            text='Select Folder',
            command = self.selectInputDirectoryTab4
        )
        selectPcapFolderButton.grid(column=0, row=1, padx=0, pady=0)
        ttk.Label(self.tab4, text= "TimeIntv (sec)").grid(column=3, row=0, padx=0, pady=0)
        self.TimeIntervalEntry_Tab4 = ttk.Entry(self.tab4,textvariable=self.TimeInterval_Tab4,width=5)
        self.TimeIntervalEntry_Tab4.grid(column=3, row=1, padx=0, pady=0)
        self.TimeInterval_Tab4.set(30)
        SelectOutputPath = ttk.Button(
        self.tab4,
        text='Select Output Path',
        command=self.selectOutputDirectoryTab4
        )
        SelectOutputPath.grid(column=4, row=1, padx=0, pady=0) 

        SelectTimeRefFile = ttk.Button(
        self.tab4,
        text='Select TimeRef',
        command=self.selectTimeRefTab4
        )
        SelectTimeRefFile.grid(column=3, row=3, padx=0, pady=0) 
        self.TimeRefFileEntry_Tab4 = ttk.Entry(self.tab4,text = "Select Time Ref File")
        self.TimeRefFileEntry_Tab4.grid(column=4, row=3, padx=0, pady=0)
        ttk.Label(self.tab4, text= "Time Ref File").grid(column=4, row=2, padx=0, pady=0)
        BatchProcess = ttk.Button(
        self.tab4,
        text='Batch Process',
        command=self.StartClipping
        )
        BatchProcess.grid(column=0, row=3, padx=0, pady=0)
        ttk.Label(self.tab4, text= "Timestamp Key").grid(column=1, row=2, padx=0, pady=0)
        self.TimeStampKeyEntry_Tab4 = ttk.Entry(self.tab4,textvariable = self.TimeStampKey_Tab4,width=15)
        self.TimeStampKeyEntry_Tab4.grid(column=1, row=3, padx=0, pady=0)
        self.TimeStampKey_Tab4.set('Timestamp')
        ttk.Label(self.tab4, text= "Timestamp Key").grid(column=1, row=2, padx=0, pady=0)
        self.FrameKeyEntry_Tab4 = ttk.Entry(self.tab4,textvariable = self.FrameKey_Tab4,width=15)
        self.FrameKeyEntry_Tab4.grid(column=2, row=3, padx=0, pady=0)
        self.FrameKey_Tab4.set('FrameIndex')
        ttk.Label(self.tab4, text= "FrameIndex").grid(column=2, row=2, padx=0, pady=0)
        



    def LoadPcap(self):
        self.ClearEvent_Tab1 = Event()
        self.LoadPcapThread = Thread(target=self.Loading_Tab1,args=(self.ClearEvent_Tab1,))
        self.LoadPcapThread.start()
        
    def ClearUp(self): # kill all thread
        self.ClearEvent_Tab1.set()
        self.FrameStatus.set("Loaded Frames:0")
        self.LoadedFrame = 0
        self.aggregated_maps = []

    def TrackerTerminate(self): # kill all thread
        self.TrackingTerminateEvent_Tab1.set()
        self.FrameStatus.set("Loaded Frames:0")
        self.TrackingStatus.set("Tracked Frames:0")

    def selectInputDirectory(self):
        filename = filedialog.askdirectory(
            title='Open a folder',
            initialdir='/',
            )
        folder_content = os.listdir(filename)
        file_types = [f.split('.')[-1] for f in folder_content]
        if 'pcap' in file_types:
            self.PcapPathEntry_Tab2.delete(0,'end')
            self.PcapPathEntry_Tab2.insert(0,filename)
        else:
            print('No Pcap in the folder')

    def selectInputDirectoryTab4(self):
        filename = filedialog.askdirectory(
            title='Open a folder',
            initialdir='/',
            )
        folder_content = os.listdir(filename)
        file_types = [f.split('.')[-1] for f in folder_content]
        if 'pcap' in file_types:
            self.PcapPathEntry_Tab4.delete(0,'end')
            self.PcapPathEntry_Tab4.insert(0,filename)
        else:
            print('No Pcap in the folder')

    def selectTrajDirectory(self):
        filename = filedialog.askdirectory(
            title='Open a folder',
            initialdir='/',
            )
        folder_content = os.listdir(filename)
        file_types = [f.split('.')[-1] for f in folder_content]
        if 'csv' in file_types:
            self.TrajPathEntry_Tab3.delete(0,'end')
            self.TrajPathEntry_Tab3.insert(0,filename)
        else:
            print('No .csv in the folder')

    def selectOutputDirectoryTab2(self):
        filename = filedialog.askdirectory(
            title='Open a folder',
            initialdir='/',
            )
        self.OutputEntry_Tab2.delete(0,'end')
        self.OutputEntry_Tab2.insert(0,filename)
    def selectOutputDirectoryTab3(self):
        filename = filedialog.askdirectory(
            title='Open a folder',
            initialdir='/',
            )
        self.OutputEntry_Tab3.delete(0,'end')
        self.OutputEntry_Tab3.insert(0,filename)
    def selectOutputDirectoryTab4(self):
        filename = filedialog.askdirectory(
            title='Open a folder',
            initialdir='/',
            )
        self.OutputEntry_Tab4.delete(0,'end')
        self.OutputEntry_Tab4.insert(0,filename)
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
        if len(filename) != 0:
            get_pcap_time(filename[0],self.PcapTime_Tab1)
    def selectRefXyzTab2(self):
        filetypes = (
            ('csv files', '*.csv'),
        )

        filename = filedialog.askopenfilenames(
            title='Open a .csv file',
            initialdir='/',
            filetypes=filetypes)

        self.RefXyzEntry_Tab2.delete(0,'end')
        self.RefXyzEntry_Tab2.insert(0,filename)
    def selectRefXyzTab3(self):
        filetypes = (
            ('csv files', '*.csv'),
        )

        filename = filedialog.askopenfilenames(
            title='Open a .csv file',
            initialdir='/',
            filetypes=filetypes)

        self.RefXyzEntry_Tab3.delete(0,'end')
        self.RefXyzEntry_Tab3.insert(0,filename)

    def selectRefLlhTab2(self):
        filetypes = (
            ('csv files', '*.csv'),
        )

        filename = filedialog.askopenfilenames(
            title='Open a .csv file',
            initialdir='/',
            filetypes=filetypes)

        self.RefLlhEntry_Tab2.delete(0,'end')
        self.RefLlhEntry_Tab2.insert(0,filename)
    def selectRefLlhTab3(self):
        filetypes = (
            ('csv files', '*.csv'),
        )

        filename = filedialog.askopenfilenames(
            title='Open a .csv file',
            initialdir='/',
            filetypes=filetypes)

        self.RefLlhEntry_Tab3.delete(0,'end')
        self.RefLlhEntry_Tab3.insert(0,filename)
    def selectTimeRefTab4(self):
        filetypes = (
            ('csv files', '*.csv'),
        )

        filename = filedialog.askopenfilenames(
            title='Open a .csv file',
            initialdir='/',
            filetypes=filetypes)

        self.TimeRefFileEntry_Tab4.delete(0,'end')
        self.TimeRefFileEntry_Tab4.insert(0,filename)
    # def selectOutput
    def Loading_Tab1(self,event):
        """
        Test Validity
        """
        if os.path.exists(self.PcapPathEntry_Tab1.get()):
            self.if_pcap_valid = True
            with open(self.PcapPathEntry_Tab1.get(), 'rb') as fpcap:
                try:
                    lidar_reader = dpkt.pcap.Reader(fpcap)
                except dpkt.dpkt.NeedData:
                    self.if_pcap_valid = False

                if self.if_pcap_valid:
                    while True:
                        try:
                            ts,buf = next(lidar_reader)
                            eth = dpkt.ethernet.Ethernet(buf)
                        except:
                            break
                        if eth.type == 2048: # for ipv4
                            if type(eth.data.data) == dpkt.udp.UDP:
                                data = eth.data.data.data
                                packet_status = eth.data.data.sport
                                if packet_status == 2368:
                                    if len(data) == 1206:
                                        self.start_timestamp = ts
                                        break
            # Loading Frames
            frame_gen = TDmapLoader(self.PcapPathEntry_Tab1.get()).frame_gen()
            while True:
                Frame = next(frame_gen) 
                if (Frame is None):
                    break 
                if event.is_set(): 
                    self.LoadedFrame = 0
                    self.aggregated_maps = []
                Td_map,Int_map = Frame
                self.LoadedFrame += 1
                self.FrameStatus.set("Loaded Frames:{}".format(len(self.aggregated_maps)))
                self.aggregated_maps.append(Td_map)
        else:
            print('Path not exists')

    def createTracker(self,event):
        self.mot = MOTLite(input_file_path='.',output_file_path='.',win_size=[self.win_size_x.get(),self.win_size_y.get()],eps = self.eps.get(),
        min_samples=self.min_samples.get(),bck_update_frame = self.bck_update_frame.get(),
        N = self.N.get(), d_thred=self.d_thred.get(),bck_n=self.bck_n.get(),
        missing_thred=self.missing_thred.get(),bck_radius = self.bck_radius.get())
        print('ss',len(self.aggregated_maps))
        self.mot.initialization(self.aggregated_maps)
        self.mot.mot_tracking(self.aggregated_maps,event,self.TrackingStatus)
        # while True:
        #     if event.is_set():
        #         break
        #     time.sleep(1)
        #     print(len(self.aggregated_maps))
        
        
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
    def Tracking(self):
        self.TrackingTerminateEvent_Tab1 = Event()
        self.TrackingThread = Thread(target=self.createTracker,args=(self.TrackingTerminateEvent_Tab1,))
        self.TrackingThread.start()
    def StartBatch(self):
        self.BatchTrackingThread = Thread(target=self.CreateBatchTracking)
        self.BatchTrackingThread.start()
    def StartGeoRef(self):
        self.GeoRefThread = Thread(target=self.CreateBatchGeoRef)
        self.GeoRefThread.start()
    def StartClipping(self):
        self.PcapClippingThread = Thread(target=self.CreateClipping)
        self.PcapClippingThread.start() 
    def CreateBatchTracking(self):

        input_path = self.PcapPathEntry_Tab2.get()
        output_traj_path = self.OutputEntry_Tab2.get()
        params = { 
                "win_size": [self.win_size_x.get(), self.win_size_y.get()], 
                "eps": self.eps.get(),
                "min_samples": self.min_samples.get(),
                "bck_update_frame":self.bck_update_frame.get(),
                "N":self.N.get(),
                "d_thred":self.d_thred.get(),
                "bck_n" : self.bck_n.get(),
                "bck_radius":self.bck_radius.get(),
                "missing_thred":self.missing_thred.get(),
                "if_save_pcd" : False}

        ref_LLH_path,ref_xyz_path = self.RefLlhEntry_Tab2.get(),self.RefXyzEntry_Tab2.get()
        ref_LLH,ref_xyz = np.array(pd.read_csv(ref_LLH_path)),np.array(pd.read_csv(ref_xyz_path))
        if len(np.unique(ref_xyz[:,2])) == 1:
            np.random.seed(1)
            offset = np.random.normal(-0.521,3.28,len(ref_LLH))
            ref_xyz[:,2] += offset
            ref_LLH[:,2] += offset * 3.2808
        ref_LLH[:,[0,1]] = ref_LLH[:,[0,1]] * np.pi/180
        ref_LLH[:,2] = ref_LLH[:,2]/3.2808
        dir_lis = os.listdir(input_path)
        traj_list = os.listdir(output_traj_path)
        pcap_names = []
        pcap_paths = []
        for f in dir_lis:
            if 'pcap' in f.split('.'):
                pcap_names.append(f)
                pcap_path = os.path.join(input_path,f)
                pcap_paths.append(pcap_path)
        print(pcap_paths)
        mots = []
        for i,p in enumerate(pcap_paths):
            f_name = pcap_names[i].split('.')[0] + '.csv'
            if f_name in traj_list:
                continue
            out_path = os.path.join(output_traj_path, f_name)
            mots.append(MOT(p,out_path,**params))
            print(out_path)
        utc_diff = self.UTC_diff.get()
        n_cpu = self.cpu_nTab2.get()
        print(f'Parallel Processing Begin with {n_cpu} Cpu(s)')
        print('Params:',params)
        p_umap(partial(run_mot,ref_LLH = ref_LLH, ref_xyz = ref_xyz, utc_diff = utc_diff), mots,num_cpus = n_cpu)
    
    def CreateBatchGeoRef(self):
        input_path = self.TrajPathEntry_Tab3.get()
        dir_lis = os.listdir(input_path)
        output_folder_path = self.OutputEntry_Tab3.get()
        csv_names = []
        for f in dir_lis:
            if 'csv' in f.split('.'):
                csv_names.append(f)
        if len(csv_names) == 0:
            print('No Trajs in Folder')
        ref_LLH_path,ref_xyz_path = self.RefLlhEntry_Tab3.get(),self.RefXyzEntry_Tab3.get()
        ref_LLH,ref_xyz = np.array(pd.read_csv(ref_LLH_path)),np.array(pd.read_csv(ref_xyz_path))
        if len(np.unique(ref_xyz[:,2])) == 1:
            np.random.seed(1)
            offset = np.random.normal(-0.521,3.28,len(ref_LLH))
            ref_xyz[:,2] += offset
            ref_LLH[:,2] += offset * 3.2808
        ref_LLH[:,[0,1]] = ref_LLH[:,[0,1]] * np.pi/180
        ref_LLH[:,2] = ref_LLH[:,2]/3.2808
        print(ref_LLH)
        print(ref_xyz)
        T = generate_T(ref_LLH,ref_xyz)
        output_folder_path = os.path.join(output_folder_path,'ProcessedTrajs')
        if os.path.exists(output_folder_path) is not True:
            os.makedirs(output_folder_path)
        n_cpu = self.cpu_nTab2.get()
        traj_paths = [os.path.join(input_path,f) for f in csv_names]
        out_paths = [os.path.join(output_folder_path,f) for f in csv_names]
        print('Begin Geo Referencing')
        p_umap(partial(run_georef,T = T), traj_paths,out_paths,num_cpus = n_cpu)

    def CreateClipping(self):
        input_path = self.PcapPathEntry_Tab4.get()
        output_path = self.OutputEntry_Tab4.get()
        timeRef = pd.read_csv(self.TimeRefFileEntry_Tab4.get())
        ts_key,frameInd_key = self.TimeStampKeyEntry_Tab4.get(),self.FrameKeyEntry_Tab4.get()
        timeintv = self.TimeInterval_Tab4.get()
        filelist = os.listdir(input_path)
        filelist_ = []
        for f in filelist:
            if len(f.split('.')) > 1:
                if f.split('.')[1] == 'pcap':
                    filelist_.append(f)
        date = [f.split('.')[0] for f in filelist_]
        date_ = []
        for d in date:
            if d[-1] == 'R':
                date_.append(d[:-2])
            else:
                date_.append(d)
        date_ = pd.to_datetime(pd.Series(date_),format=('%Y-%m-%d-%H-%M-%S'))
        Ts_records = pd.to_datetime(timeRef.loc[:,ts_key],format=('%Y-%m-%d-%H-%M-%S'))
        Pcap_inds = []
        for i in range(len(Ts_records)):
            TimeDiff = (Ts_records.iloc[i] - date_)
            within30 = (TimeDiff < pd.Timedelta(30,unit='Minute')) & ((TimeDiff >= pd.Timedelta(0,unit='Minute')))
            if within30.sum() == 0:
                Pcap_ind = -1
            else:
                Pcap_ind = TimeDiff.loc[within30].argsort().index[0]
            Pcap_inds.append(Pcap_ind)
        Pcap_inds = np.array(Pcap_inds)
        uni_ind = np.unique(Pcap_inds)
        target_frames = []
        pcap_paths = []
        pcap_names = []
        for i in uni_ind:
            if i == -1:
                continue
            start_frames = np.array(timeRef.loc[Pcap_inds==i,frameInd_key] - timeintv*10).reshape(-1,1)
            end_frames = np.array(timeRef.loc[Pcap_inds==i,frameInd_key] + timeintv*10).reshape(-1,1)
            start_frames[start_frames < 0] = 0
            end_frames[end_frames > 17999] = 17999
            target_frames.append(np.concatenate([start_frames,end_frames],axis = 1))
            pcap_paths.append(os.path.join(input_path,filelist_[i]))
            pcap_names.append(filelist_[i])
        n_cpu = self.cpu_nTab4.get()
        print('Begin Pcap Clipping with {} Cpus'.format(n_cpu))
        p_umap(partial(run_clipping,output_path = output_path), pcap_paths,target_frames,pcap_names,num_cpus = n_cpu)

if __name__ == "__main__":
    
    GUI = Interface()
    GUI.root.mainloop()



    