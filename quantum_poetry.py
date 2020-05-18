#!/usr/bin/python
# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------
# PROGRAM: worldlines.py
#-----------------------------------------------------------------------
# Version 0.1
# 18 May, 2020
# Dr Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#-----------------------------------------------------------------------
import pandas as pd
import re
from collections import Counter
#-----------------------------------------------------------------------

#-----------------------------------------------------------------------
# METHODS
#-----------------------------------------------------------------------
def extract_unique(input_file, output_file):
    """
    Write unique word list to file
    """
    z = []
    with open(input_file,'r') as fileIn, open(output_file,'w') as fileOut:
        for line in fileIn:
            for word in line.split():
                if word not in z:
                    z.append(word)
                    fileOut.write(word+'\n')
                 
def word_in_line(word, line):
    """
    Check if a word is in a line of text

    Parameters
    ----------
    word : str
    text : str

    Returns
    -------
    bool : True if word is in text, otherwise False.

    Examples
    --------
    >>> is_word_in_text("Python", "python is awesome.")
    True
    """
    pattern = r'(^|[^\w]){}([^\w]|$)'.format(word)
    pattern = re.compile(pattern, re.IGNORECASE)
    matches = re.search(pattern, text)
    return bool(matches)
#-----------------------------------------------------------------------

input_file = 'poem.txt'
output_file = 'unique_words.txt'

# Parse text and output unique words to .txt                
extract_unique(input_file, output_file)

# Store text as a single string and lines in a list
textstr = ''
lines = []
with open (input_file, 'rt') as f:      
    for line in f:                      
        lines.append(line.rstrip('\n'))
        textstr = textstr + line.rstrip('\n') + ' '

# Clean text and lower case all words
for char in '-.,\n':
    textstr = textstr.replace(char,' ')
textstr = textstr.lower() 
wordlist = textstr.split()

# Store unique words in an array
g = open(output_file, 'r')
word_list = g.readlines()
uniquewords = [x.strip() for x in word_list] 

# Word and line counts
nwords = len(wordlist)         # --> 222
nlines = len(lines)            # --> 26
nunique = len(unique_words)    # --> 140

# Word frequencies
word_freq = Counter(wordlist).most_common() # --> word_freq[0][0] = 'the' and word_freq[0][1] = '13'

print(textstr)
print(wordlist)
print(uniquewords)
print('N(words)=', nwords)
print('N(lines)=', nlines)
print('N(unique)=', nunique)

# Word index test
index = 0
prev = 0                      
str = lines[0]          

for char in '-.,\n':
    str = str.replace(char,' ')
str = str.lower() 
linewordlist = str.split()

substr = unique_words[0].lower()
index = str.find(substr)  # set index to first occurrence of substr
print(index)


#-----------------------------------------------------------------------
print('** END (initial import')

	 