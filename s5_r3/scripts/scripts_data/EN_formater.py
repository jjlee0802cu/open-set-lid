# UNI: jjl2245, me2680
"""Formats the English OpenSLR dataset"""

import csv
from os.path import exists
import sys
from pandas import read_csv
import numpy
from pathlib import Path

Path('/root/dir/sub/file.ext').stem
csv.field_size_limit(sys.maxsize) # expand the csv reader's max size
with open('../../data/EN/text.txt') as file: # Open up the text.txt file
    tsv_file = csv.reader(file, delimiter="\t") # read the file like a tsv
    for line in tsv_file: # iterate through each line
        file_name = Path(line[0]).stem # get the audio file's filepath
        if exists('../EN/' + file_name + '.wav'): # make sure the wav file exists
            text = line[-1] # get the transcription for the wav file
            with open('../EN/' + file_name + '.txt', 'w') as f: # open a new text file to save the transcription
                f.write(text) # write the transcription line to the text file