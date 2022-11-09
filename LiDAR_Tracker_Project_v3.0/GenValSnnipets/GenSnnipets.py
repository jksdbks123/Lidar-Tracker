import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
from tkinter import filedialog
import pandas as pd
import dpkt
import numpy as np
import os

def genSnippets(pcap_name,start_frames,end_frames,output_path,input_path):
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
def selectTimetable():
    filetypes = (
        ('csv files', '*.csv'),
    )

    filename = filedialog.askopenfilenames(
        title='Open a file',
        initialdir='/',
        filetypes=filetypes)
    Timetable_entry.delete(0,'end')
    Timetable_entry.insert(0,filename)
    
def selectInputDirectory():


    filename = filedialog.askdirectory(
        title='Open a folder',
        initialdir='/',
        )
    input_entry.delete(0,'end')
    input_entry.insert(0,filename)

def selectOutputDirectory():


    filename = filedialog.askdirectory(
        title='Open a folder',
        initialdir='/',
        )
    output_entry.delete(0,'end')
    output_entry.insert(0,filename)

def RunCode():
    
#     print(Timetable_path.get())
    time_interval = int(VisualizationInterval.get()) # time interval to show
    TimeTable = pd.read_csv(Timetable_path.get())
    pcap_list = os.listdir(Input_path.get())
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
        pcap_path = os.path.join(Input_path.get(),f + '.pcap')
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
        genSnippets(f,start_frames,end_frames,Output_path.get(),Input_path.get())

root = tk.Tk()
root.title('Pcap Snippets Capturing')
root.resizable(False, False)
root.geometry('700x300')
Input_path = tk.StringVar()
Output_path = tk.StringVar()
Timetable_path = tk.StringVar()
VisualizationInterval = tk.StringVar()


Interval_label = ttk.Label(root, text="TimeInterval (sec)")
Interval_label.pack(expand=True)
Interval_entry = ttk.Entry(root, textvariable = VisualizationInterval)
Interval_entry.insert(0,30)
Interval_entry.pack(expand=True)

Timetable_label = ttk.Label(root, text="Time Table Path:")
Timetable_label.pack(fill='x', expand=True)
Timetable_entry = ttk.Entry(root, textvariable=Timetable_path)
Timetable_entry.pack(fill='x', expand=True)
selectTTButton = ttk.Button(
    root,
    text='Choose Time Table Files (csv)',
    command=selectTimetable
)
selectTTButton.pack(expand=True)


input_label = ttk.Label(root, text="Input Folder:")
input_label.pack(fill='x', expand=True)
input_entry = ttk.Entry(root, textvariable=Input_path)
input_entry.pack(fill='x', expand=True)
selectInputButton = ttk.Button(
    root,
    text='Choose Input Directory',
    command=selectInputDirectory
)
selectInputButton.pack(expand=True)

output_label = ttk.Label(root, text="Output Folder:")
output_label.pack(fill='x', expand=True)
output_entry = ttk.Entry(root, textvariable=Output_path)
output_entry.pack(fill='x', expand=True)
selectOutputButton = ttk.Button(
    root,
    text='Choose Output Directory',
    command=selectOutputDirectory
)
selectOutputButton.pack(expand=True)


RunButton = ttk.Button(
    root,
    text='Run!',
    command=RunCode
)
RunButton.pack(expand=True)

root.mainloop()