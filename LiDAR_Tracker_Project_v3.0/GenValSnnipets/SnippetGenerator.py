import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter import filedialog
import pandas as pd
import dpkt
import numpy as np
import os

class SnippetGenerator():

    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Pcap Snippets Capture powered by czh')
        self.root.resizable(False, False)
        self.root.geometry('700x300')
        
        # Variables
        self.Input_path = tk.StringVar()
        self.Output_path = tk.StringVar()
        self.Timetable_path = tk.StringVar()
        self.VisualizationInterval = tk.StringVar()

        # Badges
        self.Interval_label = ttk.Label(self.root, text="TimeInterval (sec)")
        self.Interval_entry = ttk.Entry(self.root, textvariable = self.VisualizationInterval)
        self.Interval_entry.insert(0,30)
        self.Timetable_label = ttk.Label(self.root, text="Time Table Path:")
        self.Timetable_entry = ttk.Entry(self.root, textvariable=self.Timetable_path)
        self.selectTTButton = ttk.Button(
        self.root,
        text='Choose Time Table Files (csv)',
        command=self.selectTimetable
        )
        self.Input_label = ttk.Label(self.root, text="Input Folder:")
        self.Input_entry = ttk.Entry(self.root, textvariable=self.Input_path)
        self.selectInputButton = ttk.Button(
            self.root,
            text='Choose Input Directory',
            command=self.selectInputDirectory
        )
        self.Output_label = ttk.Label(self.root, text="Output Folder:")
        self.Output_entry = ttk.Entry(self.root, textvariable=self.Output_path)
        self.selectOutputButton = ttk.Button(
            self.root,
            text='Choose Output Directory',
            command=self.selectOutputDirectory
        )
        self.RunButton = ttk.Button(
            self.root,
            text='Run!',
            command=self.RunCode
        )       
        self.Interval_label.pack(expand=True)
        self.Interval_entry.pack(expand=True)
        self.Timetable_label.pack(fill='x', expand=True)
        self.Input_label.pack(fill='x', expand=True)
        self.Input_entry.pack(fill='x', expand=True)
        self.selectInputButton.pack(expand=True)
        self.Output_label.pack(fill='x', expand=True)
        self.Output_entry.pack(fill='x', expand=True)
        self.selectOutputButton.pack(expand=True)
        self.RunButton.pack(expand=True)


    def selectTimetable(self):
        filetypes = (
            ('csv files', '*.csv'),
        )

        filename = filedialog.askopenfilenames(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes)

        self.Timetable_entry.delete(0,'end')
        self.Timetable_entry.insert(0,filename)
        
    def selectOutputDirectory(self):

        filename = filedialog.askdirectory(
            title='Open a folder',
            initialdir='/',
            )
        self.Output_entry.delete(0,'end')
        self.Output_entry.insert(0,filename)

    def selectInputDirectory(self):

        filename = filedialog.askdirectory(
            title='Open a folder',
            initialdir='/',
            )
        self.Input_entry.delete(0,'end')
        self.Input_entry.insert(0,filename)
    def genSnippets(self,pcap_name,start_frames,end_frames,output_path,input_path):
        # load packets from pcap until the last frame in the end_frames
        pcap_path = os.path.join(input_path,pcap_name + '.pcap')
        packets = []
        tses = []
        frame_index = []
        cur_ind = 0
        print('Processing', pcap_name)
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
                if cur_ind > max(end_frames):
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
        result_folder_path = os.path.join(output_path,pcap_name)
        if not os.path.exists(result_folder_path):
            os.mkdir(result_folder_path)
    
        for i in range(len(start_frames)):
            with open(os.path.join(result_folder_path,'{}_{}.pcap'.format(start_frames[i],end_frames[i])),'wb') as wpcap:
                lidar_writer = dpkt.pcap.Writer(wpcap)
                start_ind = np.where(np.array(frame_index) == start_frames[i])[0][0]
                end_ind = np.where(np.array(frame_index) == end_frames[i])[0][-1]
                for f_ind in range(start_ind,end_ind):
                    lidar_writer.writepkt(packets[f_ind],ts = tses[f_ind])

    def RunCode(self):
    
    #     print(Timetable_path.get())
    
        time_interval = int(self.VisualizationInterval.get()) # time interval to show
        TimeTable = pd.read_csv(self.Timetable_path.get())
        pcap_list = os.listdir(self.Input_path.get())
        pcap_list = [f for f in pcap_list if 'pcap' in f.split('.')]
        date = [f.split('.')[0] for f in pcap_list]
        valid_date = []
        for f in TimeTable.columns:
            if f not in date:
                print('{} is not in the folder'.format(f))
            else:
                valid_date.append(f)
        for f in valid_date:
            # f -> pcap name without .pcap 
            # iterate through each pcap files in the folder and recorded in the time table 
            tses = TimeTable.loc[:,f].dropna().tolist() # [frame_ind1,frame_ind2, ... ]
            start_frames = []
            end_frames = []
            for t in tses: 
                start_frame = t - time_interval * 10
                end_frame = t + time_interval * 10
                if start_frame < 0:
                    start_frame = 0
                if end_frame > 17999:
                    end_frame = 17999
                start_frames.append(int(start_frame))
                end_frames.append(int(end_frame))
            self.genSnippets(f,start_frames,end_frames,self.Output_path.get(),self.Input_path.get())

if __name__ == "__main__":
    gen = SnippetGenerator()
    gen.root.mainloop()
