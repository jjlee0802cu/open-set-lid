# UNI: jjl2245, me2680
"""Formats the Bengali OpenSLR dataset"""

import csv
from os.path import exists
import sys
from pandas import read_csv
import numpy

csv.field_size_limit(sys.maxsize) # expand the size of the csv reader's cache
with open('../../data/BN/utt_spk_text.tsv') as file: # Bengali dataset transcription comes as a tsv file
    tsv_file = csv.reader(file, delimiter="\t") # Read the file as tsv
    for line in tsv_file: # Read each line of the tsv
        file_name = line[0] # The file name that corresponds to this line's transcription
        if exists('../BN/' + file_name + '.flac'): #make sure that the flac file exists
            text = line[-1] # grab the transcription of the flac file
            with open('../BN/' + file_name + '.txt', 'w') as f: # open up a txt file for that flac file
                f.write(text) # write that flac file's transcription to the corresponding txt file
