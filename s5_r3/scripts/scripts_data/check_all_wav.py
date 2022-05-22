# UNI: jjl2245, me2680
"""A script that goes through all files in a directory making sure that the audio files are all WAV format"""

from os import listdir
from os.path import isfile, join

path = '../'
folders = [f for f in listdir(path) if not isfile(join(path, f))] # Get all folders in a directory
for f in folders: # Go through folders
    path1 = path + f # Create a new filepath with the current working folder
    files = [f for f in listdir(path1) if isfile(join(path1, f))] # Get all files in the folder
    for f in files: # Iterating through all files
        if f.endswith('.flac'): # Check to see if the files are flac
            print("flac file found!!!") # Alert the user