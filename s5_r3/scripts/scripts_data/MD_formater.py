# UNI: jjl2245, me2680
"""Formats the Mandarin OpenSLR dataset"""

import csv
from os.path import exists
import sys
from pandas import read_csv
import numpy
import json


lines = [] # store the lines foudn in the json
for line in open('../../data/MD/set1_transcript.json', 'r'): # open up the trancript.json
    lines.append(json.loads(line)) # add to the lines array
line = lines[0] # start with the first line
for i in line: # iterate through all lines
    this_file = i['file'] # the 'file' field is the file name
    text = i['text'] # the 'text' field is the transcription
    path = '../../data/MD/audio_files/' + this_file[0] + '/' + this_file[0] + this_file[1] + '/' # create a file path for the audio file
    path_plus_wav = path + this_file # filename for wav file
    path_plus_txt = path + this_file.rsplit( ".", 1 )[ 0 ] + '.txt' # filename for the transcription
    if exists(path_plus_wav): # if the wav file exists
        with open(path_plus_txt, 'w') as f: # create a text file
            f.write(text) # write the transcription to it.
