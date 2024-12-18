import pandas as pd
import numpy as np
import os
import dpkt

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