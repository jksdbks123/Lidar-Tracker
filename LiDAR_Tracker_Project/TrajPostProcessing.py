import pandas as pd

input_path = r' '
output_path = r' '
len_thred = 10
data = pd.read_csv(r'D:\LiDAR_Data\MidTown\California\OutputFile\OutputTrajs/Trajctories_0.csv')

trajs = []
for i,t in data.groupby('ObjectID'):
    if len(t) > len_thred:
        trajs.append(t)
trajs = pd.concat(trajs)

trajs.to_csv(output_path,index = False)
