import pandas as pd
import numpy as np
import os
import dpkt
from p_tqdm import p_umap
from functools import partial

def analyze_availability(pcap_folder,ref_table,date_column_name, frame_column_name,output_name_column, time_interval):
    # ref_table: a pandas dataframe with datetime and frame index
    # date_column_name: str, the column name of the datetime
    # frame_column_name: str, the column name of the frame index
    # time_interval: int, the time interval in seconds
    # return: a list of 2D np.array, each row is a start and end frame
    # the start frame is the frame index of the first packet in the time interval
    # the end frame is the frame index of the last packet in the time interval
    # pcap naming format: Year-Month-Day-Hour-Minute-Second(-R).pcap
    pcap_list = os.listdir(pcap_folder)
    date_str = []
    for f in pcap_list:
        if f.split('.')[0].split('-')[-1] == 'R':
            date_str.append(f.split('.')[0][:-2])
        else:
            date_str.append(f.split('.')[0])
    pcap_date = pd.to_datetime(pd.Series(date_str),format=('%Y-%m-%d-%H-%M-%S'))
    query_date = pd.to_datetime(ref_table.loc[:,date_column_name],format=('%Y-%m-%d-%H-%M-%S'))
    query_frame_index = ref_table.loc[:,frame_column_name]
    output_names = ref_table.loc[:,output_name_column]

    pcap_inds = []
    for i in range(len(query_date)):
        TimeDiff = (query_date.iloc[i] - pcap_date)
        within30 = (TimeDiff < pd.Timedelta(30,unit='Minute')) & ((TimeDiff >= pd.Timedelta(0,unit='Minute')))
        valid_pcap_ind = TimeDiff.loc[within30].argsort().index[0] if within30.sum() > 0 else -1
        pcap_inds.append(valid_pcap_ind)

    pcap_inds = np.array(pcap_inds)
    uni_inds = np.unique(pcap_inds)

    target_frames = []
    pcap_paths_ = []
    output_names_ = []

    for i in uni_inds:
        if i == -1:
            continue
        start_frames = np.array(query_frame_index.loc[pcap_inds==i] - time_interval*10).reshape(-1,1)
        end_frames = np.array(query_frame_index.loc[pcap_inds==i] + time_interval*10).reshape(-1,1)
        start_frames[start_frames < 0] = 0
        end_frames[end_frames > 17999] = 17999
        target_frames.append(np.concatenate([start_frames,end_frames],axis = 1))
        pcap_paths_.append(os.path.join(pcap_folder,pcap_list[i]))
        output_names_.append(output_names.loc[pcap_inds==i].values)
    return target_frames,pcap_paths_,output_names_
        
    
    

def run_clipping(start_end_frame_list,pcap_path,output_names,output_folder):
    # load packets from pcap until the last frame in the end_frames
    # target_frame: a n x 2 np.array, with first column start frame and second column end frame
    
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
            if cur_ind > start_end_frame_list[:,1].max():
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
    for i in range(len(start_end_frame_list)):
        with open(os.path.join(output_folder,output_names[i]),'wb') as wpcap:
            lidar_writer = dpkt.pcap.Writer(wpcap)
            start_ind = np.where(frame_index == start_end_frame_list[i,0])[0][0]
            end_ind = np.where(frame_index == start_end_frame_list[i,1])[0][0]
            for f_ind in range(start_ind,end_ind):
                lidar_writer.writepkt(packets[f_ind],ts = tses[f_ind])

def run_batch_clipping(pcap_folder,output_folder,time_reference_file,
                        date_column_name,frame_column_name,time_interval,output_name_column,n_cpu):
    ref_table = pd.read_csv(time_reference_file)
    target_frames,pcap_paths_,output_names_ = analyze_availability(pcap_folder,ref_table,date_column_name,
                                                                    frame_column_name,output_name_column, 
                                                                    time_interval)
    p_umap(partial(run_clipping,output_path = output_folder), target_frames,pcap_paths_,output_names_,num_cpus = n_cpu)


if __name__ == "__main__":
    pcap_folder = r'D:\LiDAR_Data\2ndPHB'
    ref_table = pd.read_csv(r'D:\LiDAR_Data\PHB_2nd_Conflicts_FINAL.csv')
    date_column_name = 'DateTime_1'
    frame_column_name = 'FrameIndex_1'
    output_name_column = 'ConflictID'
    time_interval = 30
    analyze_availability(pcap_folder,ref_table,date_column_name, frame_column_name,output_name_column, time_interval)