import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter import filedialog
import pandas as pd
import dpkt
import numpy as np
import os
from threading import *
import time
from BfTableGenerator import TDmapLoader
from tqdm import tqdm

class Tracker():
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

        # Visulization Tab 1 
        ttk.Label(self.tab2, text="Generate Trajectories From Single Pcap File").grid(column=0, row=0, padx=30, pady=30)
        # Visulization Tab 1 
        ttk.Label(self.tab3, text="Batch Trajectories Generation").grid(column=0, row=0, padx=30, pady=30)
        self.tabControl.pack(expand = 1, fill ="both")
        self.CreateTab1()

        #Memorry
        self.aggregated_maps = []



    def CreateTab1(self):
        self.PcapPathEntry_Tab1 = ttk.Entry(self.tab1,text = "Select Pcap File")
        self.PcapPathEntry_Tab1.grid(column=1, row=1, padx=30, pady=0)
        selectInputButton = ttk.Button(
            self.tab1,
            text='Select Pcap File',
            command = self.selectPcap
        )
        selectInputButton.grid(column=0, row=1, padx=30, pady=0)
        RunButton = ttk.Button(
            self.tab1,
            text='Run!',
            command=self.RunVis
        )
        RunButton.grid(column=0, row=3, padx=30, pady=30)  

        forward_ = ttk.Button(
            self.tab1,
            text='Forward-',
            command = self.PlayForward_
        )
        forward_.grid(column=0, row=1, padx=30, pady=0)  
        
        forward = ttk.Button(
            self.tab1,
            text='Forward',
            command = self.PlayForward_
        )
        forward.grid(column=0, row=1, padx=30, pady=0)  


    def RunVis(self):
        t1 = Thread(target=self.Visulization)
        t1.start()
        t2 = Thread(target=self.LoadPcap)
        t2.start()

    def PlayForward(self):
        pass
    def PlayForward_(self):
        pass
    def PlayBackward(self):
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

    def LoadPcap(self):
        frame_gen = TDmapLoader(self.PcapPathEntry_Tab1.get()).frame_gen()
        while True:
            Frame = next(frame_gen)
            if Frame is None:
                break 
            Td_map,Int_map = Frame
            self.aggregated_maps.append(Td_map)
        

    def Visulization(self):

        while True:
            print(len(self.aggregated_maps))
            time.sleep(1)



        


if __name__ == "__main__":
    tracker = Tracker()
    tracker.root.mainloop()    


    