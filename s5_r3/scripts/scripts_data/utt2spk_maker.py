# UNI: jjl2245, me2680
"""This script creates the utt2spk file needed in each folder"""

from collections import defaultdict
import os
from os.path import exists
import re

langs = ['BN', 'EN', 'ES', 'FR', 'JV', 'KR', 'MD', 'RS', 'SW'] # the languages in our experiment
for lang in langs: # for each language
    toprint = [] # store what we will be writing to the file
    with open('../../data/' + lang + '/test/wav.scp', 'r') as text: # open the wav.scp file
        with open('../../data/' + lang + '/test/utt2spk', 'w') as utt2spk: # open a new utt2spk file to write to
            for line in text: # for each line in wav.scp
                words = line.split(' ',1)[0] # split the line into the filename and transcription and take the first
                toprint.append(words + ' ' + words + '\n') # store the filename followed by filename
            toprint = list(set(toprint)) # make it uinque and back to a list
            toprint.sort() # sort it alphabetically
            for x in toprint: # for each thing to print, 
                utt2spk.write(x) # write it to utt2spk

    toprint = [] # store what we will be writing to the file
    with open('../../data/' + lang + '/train/wav.scp', 'r') as text1: # open the wav.scp file
        with open('../../data/' + lang + '/train/utt2spk', 'w') as utt2spk: # open a new utt2spk file to write to
            for line in text1: # for each line in wav.scp
                words = line.split(' ',1)[0] # split the line into the filename and transcription and take the first
                toprint.append(words + ' ' + words + '\n') # store the filename followed by filename
            toprint = list(set(toprint)) # make it uinque and back to a list
            toprint.sort() # sort it alphabetically
            for x in toprint: # for each thing to print, 
                utt2spk.write(x) # write it to utt2spk

