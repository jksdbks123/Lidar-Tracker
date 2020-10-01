import imageio
from celluloid import Camera
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib import cm

def get_color(key):
    c_ind = int(key%len(cm.Set1.colors))
    c = cm.Set1(c_ind) 
    return np.array(c)

if __name__ == "__main__":
    print(get_color(12))