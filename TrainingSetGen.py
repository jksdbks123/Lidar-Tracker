
from MultiTrackingSystem import MultiTrackingSystem
import matplotlib.pyplot as plt
import numpy as np

class TrainingSetGen():
    def __init__(self,file_path):
        self.trackings = {}
        self.file_path = file_path
    def generate_trackings(self):
        