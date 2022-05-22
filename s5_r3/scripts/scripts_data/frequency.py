# UNI: jjl2245, me2680
"""A script to get the 1000 more used words in a language's transcriptions"""

from collections import defaultdict
import os
from os.path import exists
import re

#make sure theres a wav associated with each txt and delete txts that don't have a wav
langs = ['BN', 'EN', 'ES', 'FR', 'JV', 'KR', 'MD', 'RS', 'TR'] # the languages to check
for lang in langs: # go through each language
    for file in os.listdir('../../data/' + lang): # go through each file in the folder
        filename = os.fsdecode(file) # get the filename
        if filename.endswith(".txt"): # if it ends with a txt
            wav_name = os.path.splitext(filename)[0] + '.wav' # generate a file name that ends with wav for the txt file
            if not os.path.exists('../../data/' + lang + '/' + wav_name): # check if the wav file exists
                os.remove('../../data/' + lang + '/' + filename) # if the wav doesn't, delete the txt file

freqs = {} # store the frequencies of each word
for lang in langs: # for each language
    lang_freqs = defaultdict(int) # create a frequency dictionary
    for file in os.listdir('../' + lang): # go through each file in the directory
        filename = os.fsdecode(file) # get teh filename
        if filename.endswith(".txt"): # if it is a txt file
            with open('../../data/' + lang + '/' + filename, 'r') as text: #open the text file
                for line in text: # and read through each line of it
                    temp = '' + line # generate a temporary copy of the line
                    res = re.sub(r'[^\w\s]', '', temp).strip().lower() # strip and lower case it
                    word_list = res.strip().split(' ') # split the line into list by space
                    for word in word_list: # for each word
                        if word != '': # unless it isn't an empty character
                            lang_freqs[word] += 1 # add to the frequency dictionary
    freqs[lang] = lang_freqs # Store the entire frequency dictionary in the freqs dictionary for this language

# Now we get the 1k most frequent words
for lang in freqs: # going thorugh all languages
    lang_freqs = freqs[lang] # get the language's frequency dictionary
    most_freq = sorted(lang_freqs.items(), key=lambda x:x[1], reverse=True) # sorted the dictionary by count
    with open('../../data/_language_data/' + lang + '/' + lang + '_freq.txt', 'w') as data_file: # Open a new file for frequency
        for i in range(1000): # for the first 1000 most frequent words
            data_file.write(most_freq[i][0] + '\n') # write the word on a new line