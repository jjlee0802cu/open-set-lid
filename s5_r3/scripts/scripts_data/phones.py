# UNI: jjl2245, me2680
"""A script that creates the phones of a language"""

from collections import defaultdict
import os
from os.path import exists
import re
import pinyin

langs = ['BN', 'EN', 'ES', 'FR', 'JV', 'KR', 'MD', 'RS', 'TR'] # the languages we care about
for lang in langs: # for each language
    with open('../../data/' + lang + '/'+lang+'_freq.txt', 'r') as f: # open the freq text file
        with open('../../data/' + lang + '/lexicon.txt', 'w') as l: # open lexicon.txt
            with open('../../data/' + lang + '/nonsilence_phones.txt', 'w') as l2: # open nonsilence_phones.txt
                with open('../../data/' + lang + '/silence_phones.txt', 'w') as l3: # open silence_phones.txt
                    with open('../../data/' + lang + '/optional_silence.txt', 'w') as l4: # open optional_silence.txt
                        phones = set() # create a set of phones so we don't have repetition
                        l.write('!SIL sil\n<UNK> spn\n') # for the lexicon, write the 2 silence phones
                        for line in f: # for each line in frequency.txt
                            temp = '' + line # create a copy of the line
                            word = re.sub(r'[^\w\s]', '', temp).strip() # strip it 
                            letters = list(word) # get the letters of the word
                            if lang == 'MD': # if we are dealing with Mandarin
                                letters = list(pinyin.get(word)) # convert the line to pinyin and get each letter
                            l.write(word + ' ') # write to the lexicon (word)
                            for i in range(len(letters)): # for each letter
                                l.write(letters[i]) # write to the lexicon (letter) 
                                phones.add(letters[i]) # add to the phones set
                                if not i == len(letters) - 1: # between each phone, add a space
                                    l.write(" ") # add a space
                            l.write('\n') # write a new line
                        phones = list(phones) # turn the phones into a list
                        phones.sort() # sort them alphabetically
                        for phone in phones: # in the nonsilence phones
                            l2.write(phone + '\n') # we write the phones in order alphabetically
                        l3.write('sil\nspn\n') # for silence_phones.txt, we write the silence phones
                        l4.write('sil\n') # in optional_sielence.txt, we write sil

