import dpkt
import argparse
import pandas as pd 
import os

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='This is a program to generate .pcap snnipets')
    parser.add_argument('-i','--input', help='path to the folder contains .pcap files and Calibration folder', required=True)
    parser.add_argument('-o','--output', help='specified output path', required=True)
    parser.add_argument('-t','--timetable', help='time tables' , required=True)
    