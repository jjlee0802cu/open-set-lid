# UNI: jjl2245, me2680
"""Making the 80:20 train test split"""

import os 
from os import listdir
from os.path import isfile, join
import glob
import librosa
 
rootdir = '../../data/' # set the root directory
folders = [] # save the folders in the root directory
for file in os.listdir(rootdir): # for each file
    d = os.path.join(rootdir, file) # join the filepaths
    if os.path.isdir(d): # if the file is a directory
        folders.append(d) # save it in the folders list

# measure length of the wav files
times = {} # storing the times of each folder
for mypath in folders: # for each folder
    total_length = 0 # keep a running total
    os.chdir(mypath) 
    for file in glob.glob("*.wav"): # if we're looking at a wav file
        full_path = str(mypath) + '/' + str(file) # save the full path
        total_length += librosa.get_duration(filename=full_path) # add to the total length
    times[mypath] = total_length # save the total length in the times dictionary

# perform the 80:20 split
for mypath in folders: # for each folder
    total_length = 0 # keep a running total again
    os.chdir(mypath)
    for file in glob.glob("*.wav"): # for each wav file
        full_path = str(mypath) + '/' + str(file) # save the full path
        total_length += librosa.get_duration(filename=full_path) # add to the total length
        if total_length > times[mypath] * 0.8: # if we're over 80% of the total duration, 
            os.rename(full_path, str(mypath) + '/test/' + str(file)) # move the file to the subdirectory test
        else: # if we're still under the 80% 
            os.rename(full_path, str(mypath) + '/train/' + str(file)) # move the file to the subdirectory train

