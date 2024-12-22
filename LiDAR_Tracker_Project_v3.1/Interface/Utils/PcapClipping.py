import pandas as pd
import numpy as np
import os
import dpkt


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
        with open(os.path.join(output_folder,'{}_{}.pcap'.format(start_end_frame_list[i,0],start_end_frame_list[i,1])),'wb') as wpcap:
            lidar_writer = dpkt.pcap.Writer(wpcap)
            start_ind = np.where(frame_index == start_end_frame_list[i,0])[0][0]
            end_ind = np.where(frame_index == start_end_frame_list[i,1])[0][0]
            for f_ind in range(start_ind,end_ind):
                lidar_writer.writepkt(packets[f_ind],ts = tses[f_ind])

def run_batch_clipping(pcap_folder,output_folder,time_reference_file,
                        pcap_column,frame_column,time_interval,output_name_column):
    target_frames,pcap_paths_,output_names_ = analyze_availability(pcap_folder,ref_table,date_column_name,
                                                                    frame_column_name,output_name_column, 
                                                                    time_interval)
    
def CreateClipping(input_path,output_path,time_ref_path):
    """
    input_path: str, path to the pcap files
    output_path: str, path to the output folder
    time_ref_path: str, path to the time reference file (csv)
    """
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
    pcap_folder = r'D:\LiDAR_Data\2ndPHB'
    ref_table = pd.read_csv(r'D:\LiDAR_Data\PHB_2nd_Conflicts_FINAL.csv')
    date_column_name = 'DateTime_1'
    frame_column_name = 'FrameIndex_1'
    output_name_column = 'ConflictID'
    time_interval = 30
    analyze_availability(pcap_folder,ref_table,date_column_name, frame_column_name,output_name_column, time_interval)