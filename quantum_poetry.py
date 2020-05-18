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
import numpy as np
import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import plotly.express as px

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

def discrete_colorscale(values, colors):
    """
    values - categorical values
    colors - rgb or hex colorcodes for len(values)-1
    returns - plotly  discrete colorscale, tickvals, ticktext
    """
    
    if len(values) != len(colors)+1:
        raise ValueError('len(values) should be = len(colors)+1')
    values = sorted(values)     
    nvalues = [(v-values[0])/(values[-1]-values[0]) for v in values]  #normalized values
    colorscale = []
    for k in range(len(colors)):
        colorscale.extend([[nvalues[k], colors[k]], [nvalues[k+1], colors[k]]])        
    tickvals = [((values[k]+values[k+1])/2.0) for k in range(len(values)-1)] 
    ticktext = [f'{int(values[k])}' for k in range(len(values)-1)]
    return colorscale, tickvals, ticktext
#-----------------------------------------------------------------------

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
for word in range(len(wordfreq)-1):
    if wordfreq[word][1] > 1:
        knotlist.append(wordfreq[word][0])
    else:
        continue

# Counts
nsentences = len(sentencelist)    # --> 
nlines = len(linelist)            # --> 26
nwords = len(wordlist)            # --> 222
nunique = len(uniquewordlist)     # --> 134
nknots = len(knotlist)            # --> 30

# Branchpoint index array
maxbranches = wordfreq[0][1]
branchpointarray = np.zeros((nknots-1, maxbranches), dtype='int')
for k in range(len(knotlist)-1):  
    index = []
    for i, j in enumerate(wordlist):
        if j == knotlist[k]:
            index.append(i)            
    branchpointarray[k,0:len(index)] = index
        
print(branchpointarray)

# Define text colormap
cmap = px.colors.sequential.Viridis_r
#cmap = px.colors.sequential.Cividis_r
#cmap = px.colors.sequential.Plotly3_r
#cmap = px.colors.sequential.Magma_r
cmap_idx = np.linspace(0,len(cmap)-1, nknots, dtype=int)
colors = [cmap[i] for i in cmap_idx]
values = np.array(np.arange(len(colors)+1))
colorscale, tickvals, ticktext = discrete_colorscale(values, colors)

# Reconstruction test: colour branchpoints with knot connectivity
fig, ax = plt.subplots()
plt.plot(np.arange(0,len(wordlist)), np.zeros(len(wordlist)))
for k in range(len(knotlist)-1):  
    plt.plot(np.arange(0,len(wordlist)), np.ones(len(wordlist))*k, color='black')
    a = branchpointarray[k,:]
    vals = a[a>0]
#    plt.plot(vals, np.zeros(len(vals)))
    plt.scatter(vals, np.ones(len(vals))*k, label=knotlist[k])
plt.xlabel('word n in text')
plt.ylabel('knot k in text (>1 connection)')
plt.title('Branch Analysis Plot')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='knots', fontsize=8)
plt.savefig('branchplot.png')

text_file = open("log.txt", "w")
text_file.write('TEXT STRING: %s' % textstr)
text_file.write('\n\n')
text_file.write('SENTENCES: %s' % sentencelist)
text_file.write('\n\n')
text_file.write('LINES: %s' % linelist)
text_file.write('\n\n')
text_file.write('WORDLIST: %s' % wordlist)
text_file.write('\n\n')
text_file.write('UNIQUE WORDS: %s' % uniquewordlist)
text_file.write('\n\n')
text_file.write('KNOTS (>1 connection): %s' % knotlist)
text_file.write('\n\n')
text_file.write('N(sentences)=%s' % nsentences)
text_file.write('\n')
text_file.write('N(lines)=%s' % nlines)
text_file.write('\n')
text_file.write('N(words)=%s' % nwords)
text_file.write('\n')
text_file.write('N(unique)=%s' % nunique)
text_file.write('\n')
text_file.write('N(knots)=%s' % nknots)
text_file.close()

#print('TEXT STRING: ', textstr)
#print('SENTENCES: ', sentencelist)
#print('LINES: ', linelist)
#print('WORDLIST: ', wordlist)
#print('UNIQUE WORDS: ', uniquewordlist)
#print('KNOTS (>1 connection)', knotlist)
#print('N(sentences)=', nsentences)
#print('N(lines)=', nlines)
#print('N(words)=', nwords)
#print('N(unique)=', nunique)
#print('N(knots)=', nknots)
            
#-----------------------------------------------------------------------
print('** END (initial import')

	 