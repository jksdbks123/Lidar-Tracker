import json
	
# Data to be written
params = {
    'd':1.2,
    'thred_s':0.3,
    'N':20,
    'delta_thred' : 1e-3,
    'step':0.1,
    'win_size':(5,11),
    'eps': 1.8,
    'min_samples':15,
    'missing_thred':7,
    'ending_frame' : 2500,
    'background_update_frame':2000,
    'save_pcd' : None,
    'save_Azimuth_Laser_info' : False,
    'result_type':'merged'
}
	
# Serializing json
with open('E:\Data\Verteran\config.json', 'w') as fp:
    json.dump(params, fp)

with open('E:\Data\Verteran\config.json') as f:
    read_params = json.load(f)
    
for key in read_params.keys():
    print(read_params[key],type(read_params[key]))
