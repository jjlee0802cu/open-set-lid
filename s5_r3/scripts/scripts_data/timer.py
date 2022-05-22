# UNI: jjl2245, me2680
"""This script times the total duration of the audio files"""

import os 
from os import listdir
from os.path import isfile, join
import glob
import librosa
 
rootdir = '../../data/' # set the root directory
folders = [] # save the language folders
for file in os.listdir(rootdir): # for each item in the os dir
    d = os.path.join(rootdir, file)  # get a directory
    if os.path.isdir(d): # if it's a directory
        folders.append(d) # add it to folders

for mypath in folders: # for each folder
    total_length = 0 # keep a running total of the duration
    os.chdir(mypath) # set os.chdir
    for file in glob.glob("*.wav"): # if it's a wav file
        full_path = str(mypath) + '/' + str(file) # save the full path of the wav file
        if total_length >= 36000: # if the total duration ismore than 10 hours
            os.remove(full_path) # delete the wav file since we don't need it anymore
            text_path = os.path.splitext(full_path)[0] + '.txt' # get the text file corresponding to the wav file 
            try: # try ot delte it
                os.remove(text_path)
            except:
                continue
        else:
            total_length += librosa.get_duration(filename=full_path) # if we're under 10 hours, add to the running total.



#check each one is at most 10 hours
for mypath in folders: # for each language
    total_length = 0 # keep a running total
    os.chdir(mypath)
    for file in glob.glob("*.wav"):
        full_path = str(mypath) + '/' + str(file) 
        total_length += librosa.get_duration(filename=full_path) # add to the total duration of wav files for language
    print(mypath, total_length)