# UNI: jjl2245, me2680
"""Creates wav.scp and 'text' file"""

import os 
from os import listdir
from os.path import isfile, join, exists
import glob
from pathlib import Path

rootdir = '../../data/' # set the root directory
folders = [] # store the folders
for file in os.listdir(rootdir): # for each file in the root
    d = os.path.join(rootdir, file) # save the directory
    if os.path.isdir(d): # if it's a directory
        folders.append(d) # save it to folders

for folder in folders: # for each folder
    testf = folder + '/test/' # get the test directory
    trainf = folder + '/train/' # get the train directory
    with open(testf + 'wav.scp', 'w') as f: # open a wav.scp file to write to 
        with open(testf + 'text', 'w') as f1: # open a text file to write to
            for file in os.listdir(testf): # for each test file
                if file.endswith(".wav"): # if it is a wav file
                    f.write(Path(file).stem + ' ' + '../data/' + testf + file + '\n') # write to the wav.scp
                    if exists(folder + '/' + Path(file).stem + '.txt'):
                        with open(folder + '/' + Path(file).stem + '.txt', 'r') as f2:
                            words = f2.readlines()
                            try:
                                words = words[0]
                            except:
                                words = ' '
                            f1.write(Path(file).stem + ' ' + words + '\n') # write to teh text file
                    else:
                        os.remove(folder + '/test/' + '/' + file) # delete the file
    # repeat with the train folder
    with open(trainf + 'wav.scp', 'w') as f:
        with open(trainf + 'text', 'w') as f1:
            for file in os.listdir(trainf):
                if file.endswith(".wav"):
                    f.write(Path(file).stem + ' ' + '../data/' + trainf + file + '\n')
                    if exists(folder + '/' + Path(file).stem + '.txt'):
                        with open(folder + '/' + Path(file).stem + '.txt', 'r') as f2:
                            words = f2.readlines()
                            try:
                                words = words[0]
                            except:
                                words = ' '
                            f1.write(Path(file).stem + ' ' + words + '\n')
                    else:
                        os.remove(folder+ '/train/' + '/' + file)