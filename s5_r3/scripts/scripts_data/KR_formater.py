# UNI: jjl2245, me2680
"""Formats the Korean OpenSLR dataset"""

from os import listdir
from os.path import isfile, join, exists
import csv
import sys
from pandas import read_csv
import numpy

path = '../../data/KR' # set the KR data root directory
def get_files_folders(path): # gets all files and folders in a directory
    files = [f for f in listdir(path) if isfile(join(path, f))] # gets the files
    folders = [f for f in listdir(path) if not isfile(join(path, f))] # gets the folders
    return files, folders 

files, folders = get_files_folders(path) 
for folder in folders: # for each folder
    temp_path = path + '/' + folder # join the paths to make a new path
    temp_path = temp_path + '/' + get_files_folders(temp_path)[-1][0]
    
    in_files, in_folders = get_files_folders(temp_path) # Get the files and folders within this filepath
    text_files = [] # store the text files
    for file in in_files: # for each file
        if file.endswith('.trans.txt'): # if it ends with 'trans.txt' 
            text_files.append(file) # when store it in the text files list
    text_file = text_files[0] # we start with the first text file

    with open(temp_path + '/' + text_file) as f: # open the text file
        lines = f.readlines() # read all lines
        for line in lines: # for each line
            split = line.split(" ", 1) # split it by space but only after splitting it once
            this_file = split[0] # the first part is the file name
            text = split[-1] # the rest of it is a transcription
            if exists(temp_path + '/' + this_file + '.flac'): # if there existsa  flac file correesponding to the filename
                with open(temp_path + '/' + this_file + '.txt', 'w') as f: # open the text file corresponding to it
                    f.write(text) # write the transcription 

            



    


