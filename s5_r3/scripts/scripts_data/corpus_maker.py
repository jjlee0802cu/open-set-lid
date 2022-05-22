# UNI: jjl2245, me2680
"""A script that makes the corpus.txt files for each language"""

from collections import defaultdict
import os
from os.path import exists
import re

langs = ['AR','BN', 'EN', 'ES', 'FR', 'JV', 'KR', 'MD', 'RS', 'SW', 'TR'] # list of languages to work with
for lang in langs: # Iterating through languages
    with open('../../data/' + lang + '/local/corpus.txt', 'w') as corpus: # Opening up a corpus.txt file
        for file in os.listdir('../../data/' + lang + '/' + 'test/'): # Going through the test directory for said language
            filename = os.fsdecode(file) # get the filename
            if filename == 'text':
                with open('../../data/' + lang + '/test/' + filename, 'r') as text: # Read through each line of the text file
                    for line in text: # Read through each line of the text file
                        temp = '' + line
                        res = re.sub(r'[^\w\s]', '', temp).strip().lower() # Strip the line and make it lowercase
                        words = res.strip().split(' ',1)[1] # strip it again and split it into list by space
                        corpus.write(words + '\n') # Get the words and write it to corpus
        for file in os.listdir('../../data/' + lang + '/' + 'train/'): # Going through the train directory for said language
            filename = os.fsdecode(file)
            if filename == 'text':
                with open('../../data/' + lang + '/train/' + filename, 'r') as text: # Read through each line of the text file
                    for line in text: # Read through each line of the text file
                        temp = '' + line
                        res = re.sub(r'[^\w\s]', '', temp).strip().lower() # Strip the line and make it lowercase
                        words = res.strip().split(' ',1)[1] # strip it again and split it into list by space
                        corpus.write(words + '\n') # Get the words and write it to corpus


