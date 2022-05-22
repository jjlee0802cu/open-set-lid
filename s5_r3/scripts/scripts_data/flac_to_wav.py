# UNI: jjl2245, me2680
"""A script that converts flac files to wav files"""

import glob, os

path = '../data/AR/' # the folder to go through all audio files for
os.chdir(path) # set the working directory
for f in glob.glob("*.flac"): # go through all files that are flac files
    os.system('ffmpeg -i ' + path + f + ' ' + path + f.rsplit( ".", 1 )[ 0 ] + '.wav') # use ffmpeg to convert it to wav
    os.remove(path + f) # delete the old flac file

for f in glob.glob("*.flac"): # make sure that there are no flac files remaining
    print(f) # print the file path if there is one
