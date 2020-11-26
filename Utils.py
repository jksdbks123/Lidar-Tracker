import imageio
import numpy as np
from matplotlib import animation, cm
from matplotlib import pyplot as plt


def get_color(key):
    c_ind = int(key%len(cm.tab20.colors))
    c = cm.tab20(c_ind) 
    return np.array(c)
def visualize_single_obj(detected_obj):
    pass

if __name__ == "__main__":
    print(get_color(12))
