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

# Store text as a single string and lines in a list
textstr = ''
linelist = []
with open (input_file, 'rt') as f:      
    for line in f:   
        if len(line)>1: # ignore empty lines                 
            linelist.append(line.rstrip('\n'))
            textstr = textstr + line.rstrip('\n')
        else:
            continue

# extract sentences into list
sentencelist = textstr.split('.')
sentencelist = sentencelist[0:-1] # ignore last entry which is '' due to final full stop

# Clean text and lower case all words
str = textstr
for char in '-.,\n':
    str = str.replace(char,' ')
str = str.lower() 
wordlist = str.split()

# Store unique words in an array
uniquewordlist = []
for word in wordlist:           
    if word not in uniquewordlist:
        uniquewordlist.append(word)
                    
# Word frequencies
wordfreq = Counter(wordlist).most_common() # --> wordfreq[0][0] = 'the' and wordfreq[0][1] = '13'

# Find knots having word frequency > 1
knotlist = []
for i in range(len(wordfreq)):
    if wordfreq[i][1] > 1:
        knotlist.append(wordfreq[i][0])
    else:
        continue

# Counts
nsentences = len(sentencelist)    # --> 
nlines = len(linelist)            # --> 26
nwords = len(wordlist)            # --> 222
nunique = len(uniquewordlist)     # --> 134
nknots = len(knotlist)           # --> 30

print(textstr)
print(sentencelist)
print(linelist)
print(wordlist)
print(uniquewordlist)
print(knotlist)

print('N(sentences)=', nsentences)
print('N(lines)=', nlines)
print('N(words)=', nwords)
print('N(unique)=', nunique)
print('N(knots)=', nknots)

# substr = uniquewords[0].lower()
# index = str.find(substr)  # set index to first occurrence of substr


#-----------------------------------------------------------------------
print('** END (initial import')

	 