# UNI: jjl2245, me2680
"""Formats the Russian OpenSLR dataset"""

import csv
from os.path import exists
import sys
from pandas import read_csv
import numpy
import json


lines = [] # store the lines
for line in open('../../data/RS/train/manifest.json', 'r'): # open up the manifest json file
    lines.append(json.loads(line)) # store them in lines array

for line in lines: # for each line
    file_path = line['audio_filepath'] # grab the file path through the audio_filepath key
    text = line['text'] # get the transcript thorugh the text key
    if exists('../../data/RS/train/' + file_path): # if we find that the file path exists, then we proceed
        with open('../../data/RS/train/' + file_path.rsplit( ".", 1 )[ 0 ] + '.txt', 'w') as f: # open a txt file for the transcript
            f.write(text) # write the transcript into the text file
