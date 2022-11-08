import dpkt
import argparse
import pandas as pd 
import os

def genSnippets(pcap_name,start_frames,end_frames,input_path):
    # load packets from pcap until the last frame in the end_frames
    pcap_path = os.path.join(input_path,f + '.pcap')
    packets = []
    tses = []
    frame_index = []
    cur_ind = 0
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
    result_folder_path = os.path.join(input_path,pcap_name)
    if os.path.exists(result_folder_path):
        os.mkdir(result_folder_path)
    
    for i in range(len(start_frames)):
        with open(os.path.join(result_folder_path,'{}_{}.pcap'.format(start_frames[i],end_frames[i])),'wb') as wpcap:
            lidar_writer = dpkt.pcap.Writer(wpcap)
            start_ind = np.where(np.array(frame_index) == start_frames[i])[0][0]
            end_ind = np.where(np.array(frame_index) == end_frames[i])[0][-1]
            for f_ind in range(start_ind,end_ind):
                lidar_writer.writepkt(packets[f_ind],ts = tses[f_ind])


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='This is a program to generate .pcap snnipets')
    parser.add_argument('-i','--input', help='path to the folder contains .pcap files and Calibration folder', required=True)
    parser.add_argument('-o','--output', help='specified output path', required=True)
    parser.add_argument('-t','--timetable', help='time tables' , required=True)
    
    input_path = 'D:\Test\TEST'
    time_interval = 30 # time interval to show
    TimeTable = pd.read_csv(os.path.join(input_path,'TimeTable.csv'))
    pcap_list = os.listdir(input_path)
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
        pcap_path = os.path.join(input_path,f + '.pcap')
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
        genSnippets(f,start_frames,end_frames,input_path)
        