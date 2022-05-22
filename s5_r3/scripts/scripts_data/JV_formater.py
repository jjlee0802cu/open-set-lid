# UNI: jjl2245, me2680
"""Formats the Javanese OpenSLR dataset"""

import csv
from os.path import exists
import sys
from pandas import read_csv
import numpy

csv.field_size_limit(sys.maxsize) # expand the csv reader's max size
with open('../../data/JV/utt_spk_text.tsv') as file: # Open up the tsv file
    tsv_file = csv.reader(file, delimiter="\t") # read the file like a tsv
    for line in tsv_file: # iterate through each line
        file_name = line[0] # get the audio file's filepath
        if exists('../JV/' + file_name + '.flac'):  # make sure the flac file exists
            text = line[-1] # get the transcription for the wav file
            with open('../JV/' + file_name + '.txt', 'w') as f: # open a new text file to save the transcription
                f.write(text) # write the transcription line to the text file
