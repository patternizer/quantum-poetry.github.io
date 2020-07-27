#! /usr/bin/ python

# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: worldlines.py
#------------------------------------------------------------------------------
# Version 0.11
# 9 July, 2020
# Dr Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS
#------------------------------------------------------------------------------
generate_anyons = True
generate_variants = True
generate_networkx_edges = True
generate_qubits = False
generate_erdos_parameter = False
generate_erdos_equivalence = False
generate_adjacency = False
qubit_logic = False
plot_branchpoint_table = True
plot_networkx_connections = True
plot_networkx_non_circular = True
plot_networkx_erdos_parameter = False
plot_networkx_erdos_equivalence = False
plot_networkx_connections_branchpoints = True
plot_networkx_connections_dags = True
plot_variants = True
machine_learning = False
write_log = True
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import scipy as sp
# import math
# math.log(N,2) for entropy calculations
import random
from random import randint
from random import randrange
# Text Parsing libraries:
import re
from collections import Counter
# Network Graph libraries:
import networkx as nx
from networkx.algorithms import approximation as aprx
# Plotting libraries:
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors as mcol
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from skimage import io
import glob
from PIL import Image
# Silence library version notifications
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# NLP Libraries
# ML Libraries
# App Deployment Libraries
# import dash
# import dash_core_components as dcc
# import dash_html_components as html
# import dash_bootstrap_components as dbc
# from dash.dependencies import Input, Output, State
# from flask import Flask
# import json
# import os
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# METHODS
#------------------------------------------------------------------------------
def word_in_line(word, line):
    """
    Check if word is in line
    word, line - str
    returns    - True if word in line, False if not
    """

    pattern = r'(^|[^\w]){}([^\w]|$)'.format(word)
    pattern = re.compile(pattern, re.IGNORECASE)
    matches = re.search(pattern, text)

    return bool(matches)

def discrete_colorscale(values, colors):
    """
    values  - categorical values
    colors  - rgb or hex colorcodes for len(values)-1
    eeturn  - discrete colorscale, tickvals, ticktext
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

def rgb2hex(colorin):
    """
    Convert (r,g,b) to hex
    """

    r = int(colorin.split('(')[1].split(')')[0].split(',')[0])
    g = int(colorin.split('(')[1].split(')')[0].split(',')[1])
    b = int(colorin.split('(')[1].split(')')[0].split(',')[2])

    return "#{:02x}{:02x}{:02x}".format(r,g,b)

def parse_poem(input_file):
    """
    Text parsing of poem and construction of branchpoint array
    """

    print('parsing poem ...')

    # Store lines in a list

    linelist = []
    with open (input_file, 'rt') as f:      
        for line in f:   
            if len(line)>1: # ignore empty lines                  
                linelist.append(line.strip())   
            else:
                continue

    # Store text as a single string

    textstr = ''
    for i in range(len(linelist)):
        if i < len(linelist) - 1:
            textstr = textstr + linelist[i] + ' '
        else:
            textstr = textstr + linelist[i]
        
    # extract sentences into list 
    # (ignore last entry which is '' due to final full stop)
            
    sentencelist = textstr.split('.')[0:-1] 

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

    # Find branchpoints having word frequency > 1

    branchpointlist = []
    for word in range(len(wordfreq)-1):
        if wordfreq[word][1] > 1:
            branchpointlist.append(wordfreq[word][0])
        else:
            continue

    # Branchpoint index array

    maxbranches = wordfreq[0][1]
    branchpointarray = np.zeros((len(branchpointlist), maxbranches), dtype='int')
    for k in range(len(branchpointlist)):  
        index = []
        for i, j in enumerate(wordlist):
            if j == branchpointlist[k]:
                index.append(i)            
        branchpointarray[k,0:len(index)] = index
    
    # Filter out multiple branchpoint in single line only occurences
    # using word indices of branchpoints and line start and end indices

    lineindices = []    
    wordcount = 0
    for i in range(len(linelist)):
        linelen = len(linelist[i].split())
        lineindices.append([i, wordcount, wordcount+linelen-1])
        wordcount += linelen
                    
    mask = []
    branchlinearray = []        
    for i in range(np.size(branchpointarray, axis=0)): # i.e. nbranchpoints        
        branchpointindices = branchpointarray[i,:][branchpointarray[i,:]>0]
        linecounter = 0 
        for j in range(len(linelist)):                     
            branchpointcounter = 0
            for k in range(len(branchpointindices)):
                if branchpointindices[k] in np.arange(lineindices[j][1],lineindices[j][2]+1):
                    branchpointcounter += 1
                    branchlinearray.append([j,i,lineindices[j][1],branchpointindices[k],lineindices[j][2]])            
            if branchpointcounter > 0:
                linecounter += 1                    
        if linecounter < 2:
            mask.append(i)            

    a = np.array(branchpointarray)
    b = branchpointlist
    for i in range(len(mask)):
        a = np.delete(a,mask[i]-i,0)        
        b = np.delete(b,mask[i]-i,0)        
    branchpointarray = a
    branchpointlist = list(b)
  
    db = pd.DataFrame(branchpointarray)
    db.to_csv('branchpointarray.csv', sep=',', index=False, header=False, encoding='utf-8')

    return textstr, sentencelist, linelist, wordlist, uniquewordlist, wordfreq, branchpointlist, branchpointarray

def generate_branchpoint_colormap(wordfreq, nbranchpoints, nwords, branchpointarray):
    """
    Generate colormap using hexcolors for all branchpoints
    """

    print('generating branchpoint_colormap ...')

    freq = [ wordfreq[i][1] for i in range(len(wordfreq)) ]
    nlabels = nbranchpoints
    cmap = px.colors.diverging.Spectral
    cmap_idx = np.linspace(0,len(cmap)-1, nlabels, dtype=int)
    colors = [cmap[i] for i in cmap_idx]
    hexcolors = [ rgb2hex(colors[i]) for i in range(len(colors)) ]
    
    branchpoint_colormap = []
    for k in range(nwords):    
        branchpoint_colormap.append('lightgrey')              

    for j in range(np.size(branchpointarray, axis=0)): # i.e. nbranchpoints            
        for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
            branchpoint_colormap[branchpointarray[j,i]] = hexcolors[j] 

    return  branchpoint_colormap, hexcolors

def compute_networkx_edges(nwords, wordlist, branchpointarray):
            
    print('computing_networkx_edges ...')
    
    # Construct edgelist, labellist
    
    edgelist = [(i,i+1) for i in range(nwords-1)]
    labellist = [{i : wordlist[i]} for i in range(nwords)]

    df = pd.DataFrame()
    
    G = nx.Graph()
    G.add_edges_from(edgelist)
    for node in G.nodes():
        G.nodes[node]['label'] = labellist[node]

    edge_colormap = []
    for k in range(nwords-1):
        edge_colormap.append('lightgrey')              
        
    for j in range(np.size(branchpointarray, axis=0)): # i.e. nbranchpoints        
        branchpointedges = []
        for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
            branchpointindices = branchpointarray[j,:]
            connections = branchpointindices[(branchpointindices != branchpointindices[i]) & (branchpointindices > 0)]
            for k in range(len(connections)):
                if branchpointindices[i] > 0:
                    branchpointedges.append([branchpointindices[i], connections[k]])
        G.add_edges_from(branchpointedges)        

#        for l in range(int(len(branchpointedges)/2)): # NB 2-driectional edges
#            edge_colormap.append(hexcolors[j])
    nedges = len(G.edges)

    # Generate non-circular form of the networkx graph

    N = nx.Graph()
    N.add_edges_from(edgelist)
    for j in range(np.size(branchpointarray, axis=0)): # i.e. nbranchpoints        
        branchpointedges = []
        for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
            branchpointindices = branchpointarray[j,:]
            connections = branchpointindices[(branchpointindices != branchpointindices[i]) & (branchpointindices > 0)]
            for k in range(len(connections)):
                if branchpointindices[i] > 0:
                    branchpointedges.append([branchpointindices[i], connections[k]])
        N.add_edges_from(branchpointedges)        
    N.remove_edges_from(edgelist)
    N_degrees = [degree for node,degree in dict(N.degree()).items()] # degree of nodes
    notbranchpoints = [ node for node,degree in dict(N.degree()).items() if degree == 0 ] # each node in circular graph has 2 neighbours at start
            
    return nedges, notbranchpoints, G, N
    
def compute_erdos_parameter(nwords, nedges):
    """
    Compute Erdos-Renyi parameter estimate
    """
    
    print('computing_erdos_parameter ...')

    edgelist = [(i,i+1) for i in range(nwords-1)]
    for connectivity in np.linspace(0,1,1000001):
        random.seed(42)
        E = nx.erdos_renyi_graph(nwords, connectivity)
        erdosedges = len(E.edges)
        if erdosedges == (nedges-len(edgelist)):            
#            print("{0:.6f}".format(connectivity))
#            print("{0:.6f}".format(erdosedges))
            nerdosedges = len(E.edges)
            return nerdosedges, connectivity, E
#            break
    nerdosedges = len(E.edges)

    return nerdosedges, connectivity, E

def compute_erdos_equivalence(nwords, nedges, N, notbranchpoints):
    """
    Compute Erdos-Renyi equivalence probability
    """

    print('computing_erdos_equivalence ...')

    # Compare Erdos-Renyi graph edges in reduced networks (branchpoint network)

    N.remove_nodes_from(notbranchpoints)
    mapping = { np.array(N.nodes)[i]:i for i in range(len(N.nodes)) }
    H = nx.relabel_nodes(N,mapping)                
    maxdiff = len(H.edges)
    iterations = 100000
    for i in range(iterations+1):

        E = nx.erdos_renyi_graph(len(H.nodes), connectivity)
        diff = H.edges - E.edges        
        if len(diff) < maxdiff:
            maxdiff = len(diff)
            commonedges = H.edges - diff      
            pEquivalence = i/iterations
            Equivalence = E
            
    return commonedges, pEquivalence, Equivalence

def compute_anyons(linelist, wordlist, branchpointarray):
    """
    Anyon construction: braiding
    """

    print('generating_anyons ...')

    # Compute start and end word indices for each line of the poem

    lineindices = []    
    wordcount = 0
    for i in range(len(linelist)):
        linelen = len(linelist[i].split())
        lineindices.append([i, wordcount, wordcount+linelen-1])
        wordcount += linelen
                    
    # For each branchpoint find line index and word indices of line start, branchpoint and line end
    # branchlinearray: [line, branchpoint, wordstart, wordbranchpoint, wordend] 
    
    branchlinearray = []        
    for i in range(np.size(branchpointarray, axis=0)): # i.e. nbranchpoints        
        branchpointindices = branchpointarray[i,:][branchpointarray[i,:]>0]
        for j in range(len(linelist)):                     
            for k in range(len(branchpointindices)):
                if branchpointindices[k] in np.arange(lineindices[j][1],lineindices[j][2]+1):                   
                    branchlinearray.append([j,i,lineindices[j][1],branchpointindices[k],lineindices[j][2]])
    
    # Filter out multiple branchpoint in single line only occurences

    a = np.array(branchlinearray)
    mask = []
    for i in range(len(branchlinearray)-2):
        if (a[i,0] == a[i+1,0]) & (a[i,1] == a[i+1,1]) & (a[i+2,1]!=a[i,1]):
            mask.append(i)
            mask.append(i+1)
    for i in range(len(mask)):
        a = np.delete(a,mask[i]-i,0)        
    branchlinearray = a[a[:,0].argsort()]

    # Filter out start of line and end of line occurring branchpoints

    a = np.array(branchlinearray)
    mask = []
    for i in range(len(branchlinearray)):
        if ((a[i,2] == a[i,3]) | (a[i,3] == a[i,4])):
            mask.append(i)
    for i in range(len(mask)):
        a = np.delete(a,mask[i]-i,0)        
    branchlinearray = a[a[:,0].argsort()]

    # Anyons
    
    anyonarray = []
    for i in range(len(linelist)):
        a = branchlinearray[branchlinearray[:,0]==i] 
        if len(a) == 0:
            break
        for j in range(len(a)):    
            anyon_pre = wordlist[a[j,2]:a[j,3]+1]
            b = branchlinearray[(branchlinearray[:,1]==a[j,1]) & (branchlinearray[:,0]!=a[j,0])]             

            #######################################################
            # For > 1 swaps, add additional anyon segment code here
            # + consider case of forward in 'time' constraint
            # + consider return to start line occurrence
            #######################################################

            if len(b) == 0:
                break
            for k in range(len(b)):
                anyon_post = wordlist[b[k,3]+1:b[k,4]+1]
                anyon = anyon_pre + anyon_post
                anyonarray.append( [i, b[k,0], branchpointlist[a[j,1]], anyon, a[j,2], a[j,3], a[j,4] ])

    df = pd.DataFrame(anyonarray)
    df.to_csv('anyonarray.csv', sep=',', index=False, header=False, encoding='utf-8')

    return anyonarray
    
def compute_variants(linelist, anyonarray):
    """
    Variant construction
    """

    print('generating_variants ...')

    # generate variants of the poem
    
    df = pd.DataFrame(anyonarray)

    allpoemsidx = []
    allpoems = []
    allidx = []
    nvariants = 0

    for i in range(len(linelist)):      
                
        a = df[df[0]==i]
        for j in range(len(a)):

            poem = []
            lineidx = []    
            lines = np.arange(len(linelist))    
        
            while len(lines)>0:
        
                print(nvariants,i,j)
                
                if len(lines) == len(linelist):
                    linestart = a[0].values[j]
                    lineend = a[1].values[j]
                    branchpoint = a[2].values[j]                    
                else:
                    b = df[df[0]==lines[0]]
                    linestart = b[0].values[0]                
                    lineend = np.setdiff1d( np.unique(b[1].values), lineidx )[0]   
                    branchpoint = df[ (df[0]==linestart) & (df[1]==lineend) ][2].values[0]
                    
                lineidx.append(linestart)    
                lineidx.append(lineend)
                branchpointstartpre = df[ (df[0]==linestart) & (df[1]==lineend) & (df[2]==branchpoint) ][4].values[0]
                branchpointstart = df[ (df[0]==linestart) & (df[1]==lineend) & (df[2]==branchpoint) ][5].values[0]
                branchpointstartpro = df[ (df[0]==linestart) & (df[1]==lineend) & (df[2]==branchpoint) ][6].values[0]
                branchpointendpre = df[ (df[0]==lineend) & (df[1]==linestart) & (df[2]==branchpoint) ][4].values[0]
                branchpointend = df[ (df[0]==lineend) & (df[1]==linestart) & (df[2]==branchpoint) ][5].values[0]
                branchpointendpro = df[ (df[0]==lineend) & (df[1]==linestart) & (df[2]==branchpoint) ][6].values[0]                
                allidx.append([nvariants, linestart, lineend, branchpoint, branchpointstartpre, branchpointstart, branchpointstartpro])
                allidx.append([nvariants, lineend, linestart, branchpoint, branchpointendpre, branchpointend, branchpointendpro])
                poem.append(df[ (df[0]==linestart) & (df[1]==lineend) & (df[2]==branchpoint) ][3].values[0])
                poem.append(df[ (df[0]==lineend) & (df[1]==linestart) & (df[2]==branchpoint) ][3].values[0])
                lines = np.setdiff1d(lines,lineidx)   
         
            nvariants += 1                                         
            poemsorted = []
            for k in range(len(lineidx)):
                poemsorted.append(poem[lineidx.index(k)])
            allpoems.append(poemsorted)
            allpoemsidx.append(lineidx)                 
            dp = pd.DataFrame(poemsorted)
            dp.to_csv('poem'+'_'+"{0:.0f}".format(nvariants-1).zfill(3)+'.csv', sep=',', index=False, header=False, encoding='utf-8')
 
    di = pd.DataFrame(allpoemsidx)    
    di.to_csv('poem_allidx.csv', sep=',', index=False, header=False, encoding='utf-8')
    da = pd.DataFrame(allpoems)
    da.to_csv('poem_all.csv', sep=',', index=False, header=False, encoding='utf-8')
    dl = pd.DataFrame(allidx)
    dl.to_csv('allidx.csv', sep=',', index=False, header=False, encoding='utf-8')

    return nvariants, allpoemsidx, allpoems, allidx

def generate_qubits():
    """
    Qubit contruction
    """    

    print('generating_qubits ...')

def qubit_logic():
    """
    Apply gates to Bell states
    """    

    print('applying logic gates ...')

def machine_learning():
    """
    Feature extraction
    """    

    print('extracting features ...')
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# LOAD POEM
#------------------------------------------------------------------------------
"""
Poem to generate quantum variants from
"""
#input_file = 'poem.txt'
input_file = 'poem-v1.txt'

textstr, sentencelist, linelist, wordlist, uniquewordlist, wordfreq, branchpointlist, branchpointarray = parse_poem(input_file)

# Counts
        
nsentences = len(sentencelist)          # --> 4
nlines = len(linelist)                  # --> 8
nwords = len(wordlist)                  # --> 98
nunique = len(uniquewordlist)           # --> 59
nbranchpoints = len(branchpointlist)    # --> 20

if generate_networkx_edges == True:
    nedges, notbranchpoints, G, N = compute_networkx_edges(nwords, wordlist, branchpointarray)
if generate_anyons == True:
    anyonarray = compute_anyons(linelist, wordlist, branchpointarray)
if generate_variants == True:
    nvariants, allpoemsidx, allpoems, allidx = compute_variants(linelist, anyonarray)
if generate_qubits == True:
    print('generating_qubits ...')
if generate_erdos_parameter == True:
    nerdosedges, connectivity, E = compute_erdos_parameter(nwords, nedges)
if generate_erdos_equivalence == True:
    commonedges, pEquivalence, Equivalence = compute_erdos_equivalence(nwords, nedges, N, notbranchpoints)
if qubit_logic == True:
    print('applying logic gates ...')
if machine_learning == True:
    print('extracting features ...')
     
# -----------------------------------------------------------------------------
branchpoint_colormap, hexcolors = generate_branchpoint_colormap(wordfreq, nbranchpoints, nwords, branchpointarray)
# -----------------------------------------------------------------------------

if plot_branchpoint_table == True:
    
    print('plotting_branchpoint_table ...')

    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot(np.arange(0,len(wordlist)), np.zeros(len(wordlist)))
    for k in range(len(branchpointlist)):  
        plt.plot(np.arange(0,len(wordlist)), np.ones(len(wordlist))*k, color='black')
        a = branchpointarray[k,:]
        vals = a[a>0]
        plt.scatter(vals, np.ones(len(vals))*k, label=branchpointlist[k], s=100, facecolors=hexcolors[k], edgecolors='black')                

    xticks = np.arange(0, len(wordlist)+0, step=10)
    xlabels = np.array(np.arange(0, len(wordlist), step=10).astype('str'))
    yticks = np.arange(0, len(branchpointlist), step=1)
    ylabels = np.array(np.arange(0, len(branchpointlist), step=1).astype('str'))
    plt.xticks(ticks=xticks, labels=xlabels)  # Set label locations
    plt.yticks(ticks=yticks, labels=ylabels)  # Set label locations
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('word n in text', fontsize=20)
    plt.ylabel('branchpoint k in text (>1 connection)', fontsize=20)
    plt.title('Branch Analysis Plot', fontsize=20)
    plt.gca().invert_yaxis()    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.savefig('branchplot.png')
    plt.close(fig)

if plot_networkx_connections == True:

    print('plotting_networkx_connections ...')
    
    fig, ax = plt.subplots(figsize=(15,10))    
    nx.draw_circular(G, node_color=branchpoint_colormap, node_size=300, linewidths=0.5, font_size=12, font_weight='normal', with_labels=True)
    plt.title('Networkx (circularly connected): N(edges)=' + "{0:.0f}".format(len(G.edges)), fontsize=20)
    plt.savefig('networkx.png')
    plt.close(fig)

if plot_networkx_non_circular == True:

    print('plotting_networkx_non_circular ...')

    fig, ax = plt.subplots(figsize=(15,10))
    nx.draw_circular(N, node_color=branchpoint_colormap, node_size=300, linewidths=0.5, font_size=12, font_weight='normal', with_labels=True)
    plt.title('Networkx (non-circularly connected): N(edges)=' + "{0:.0f}".format(len(N.edges)), fontsize=20)
    plt.savefig('networkx_non_circular.png')

if plot_networkx_erdos_parameter == True:
    
    print('plotting_networkx_erdos ...')

    fig, ax = plt.subplots(figsize=(15,10))
    nx.draw_circular(E, node_color=branchpoint_colormap, node_size=300, linewidths=0.5, font_size=12, font_weight='normal', with_labels=True)
    plt.title('Erdős-Rényi Model: p=' + "{0:.6f}".format(connectivity) + ', N(edges)=' + "{0:.0f}".format(nerdosedges), fontsize=20)
    plt.savefig('networkx_erdos.png')
    plt.close(fig)
        
if plot_networkx_erdos_equivalence == True:
    
    print('plotting_networkx_erdos_equivalence ...')

    fig, ax = plt.subplots(figsize=(15,10))
    nx.draw_circular(Eequivalence, node_color='lightgrey', node_size=300, linewidths=0.5, font_size=12, font_weight='normal', with_labels=True)
    plt.title('Erdős-Rényi Model (equivalent): N(common edges)=' + "{0:.0f}".format(len(N.edges)-len(diff)), fontsize=20)
    plt.savefig('networkx_erdos_equivalence.png')

if plot_variants == True:

    print('plotting_variants ...')
    
    di = pd.DataFrame(allpoemsidx)
    da = pd.DataFrame(allpoems)
    dl = pd.DataFrame(allidx)

    for i in range(nvariants):
          
        connectorstart = []
        connectorend = []

        fig, ax = plt.subplots(figsize=(15,10))
        for k in range(len(linelist)):  
            
            branchpoint = dl[(dl[0]==i)&(dl[1]==k)][3].values[0]                
            linestart = dl[(dl[0]==i)&(dl[1]==k)][1].values[0]
            lineend = dl[(dl[0]==i)&(dl[1]==k)][2].values[0]
            plt.scatter(np.arange(0,len(linelist[k].split())), np.ones(len(linelist[k].split()))*k, color='black')
            if linestart < lineend:
                x1 = np.arange(0, dl[(dl[0]==i)&(dl[1]==k)][5].values[0] - dl[(dl[0]==i)&(dl[1]==k)][4].values[0]+1)
                x2 = np.arange(dl[(dl[0]==i)&(dl[1]==k)][5].values[0] - dl[(dl[0]==i)&(dl[1]==k)][4].values[0]+1, dl[(dl[0]==i)&(dl[1]==k)][6].values[0]-dl[(dl[0]==i)&(dl[1]==k)][4].values[0]+1)                               
                y1 = np.ones(len(x1))*k
                y2 = np.ones(len(x2))*k    
                plt.plot(x1,y1,'blue')
                plt.plot(x2,y2,'red')
                plt.scatter(x1[-1], y1[-1], s=100, facecolors=hexcolors[branchpointlist.index(branchpoint)], edgecolors='black')      
                connectorstart.append([linestart, x1[-1], y1[-1]])                
                connectorend.append([lineend, x2[0], y2[0]])     
            else:
                x3 = np.arange(dl[(dl[0]==i)&(dl[1]==k)][5].values[0] - dl[(dl[0]==i)&(dl[1]==k)][4].values[0]+1, dl[(dl[0]==i)&(dl[1]==k)][6].values[0]-dl[(dl[0]==i)&(dl[1]==k)][4].values[0]+1)                               
                x4 = np.arange(0, dl[(dl[0]==i)&(dl[1]==k)][5].values[0] - dl[(dl[0]==i)&(dl[1]==k)][4].values[0]+1)               
                y3 = np.ones(len(x3))*k
                y4 = np.ones(len(x4))*k
                plt.plot(x3,y3,'blue')
                plt.plot(x4,y4,'red')       
                plt.scatter(x4[-1], y4[-1], s=100, facecolors=hexcolors[branchpointlist.index(branchpoint)], edgecolors='black')                
                connectorstart.append([linestart, x3[0], y3[0]])                
                connectorend.append([lineend, x4[-1], y4[-1]])     

        for k in range(len(linelist)):  
            
            branchpoint = dl[(dl[0]==i)&(dl[1]==k)][3].values[0]                
            linestart = dl[(dl[0]==i)&(dl[1]==k)][1].values[0]
            lineend = dl[(dl[0]==i)&(dl[1]==k)][2].values[0]
            print(k, linestart, lineend)
            if linestart < lineend:
                x1 = connectorstart[linestart][1]
                y1 = connectorstart[linestart][2]
                x2 = connectorend[lineend][1]+1
                y2 = connectorend[lineend][2]
                x = [x1,x2]
                y = [y1,y2]                
                plt.plot(x,y,'blue')
            else:
                x1 = connectorend[lineend][1]
                y1 = connectorend[lineend][2]
                x2 = connectorstart[linestart][1]-1
                y2 = connectorstart[linestart][2]
                x = [x1,x2]
                y = [y1,y2]                
                plt.plot(x,y,'red')
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('word in anyon', fontsize=20)
        plt.ylabel('line in text', fontsize=20)
        plt.title('Anyon Plot for variant: ' + i.__str__(), fontsize=20)
        plt.gca().invert_yaxis()    
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        plt.savefig('variant_anyons_' + i.__str__().zfill(3) +'.png')        
        plt.close(fig)
    
    # Generate animated GIF

    fp_in = "variant_anyons_*.png"
    fp_out = "variant_anyons.gif"

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
#    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in), key=os.path.getmtime)]    
    img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=1000, loop=0)
    
if plot_networkx_connections_branchpoints == True:
    
    print('plotting_networkx_connections_branchpoints ...')
    
    # Construct edgelist, labellist
    
    edgelist = [(i,i+1) for i in range(nwords-1)]
    labellist = [{i : wordlist[i]} for i in range(nwords)]

    for j in range(np.size(branchpointarray, axis=0)): # i.e. nbranchpoints        
        
    # Plot wordfreq colour-coded networkx graph of connectivity
     
        fig, ax = plt.subplots(figsize=(15,10))    
        G = nx.DiGraph()
        G.add_edges_from(edgelist)
        for node in G.nodes():
            G.nodes[node]['label'] = labellist[node]

        branchpointedges = []
        for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
            branchpointindices = branchpointarray[j,:]
            connections = branchpointindices[(branchpointindices != branchpointindices[i]) & (branchpointindices > 0)]
            for k in range(len(connections)):
                if branchpointindices[i] > 0:
                    branchpointedges.append([branchpointindices[i], connections[k]])
        G.add_edges_from(branchpointedges)

        # Colormap per branchpoint
    
        colormap = []
        for k in range(nwords):
            colormap.append('lightgrey')              
        for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
            branchpointindices = branchpointarray[j,:]
            connections = branchpointindices[(branchpointindices != branchpointindices[i]) & (branchpointindices > 0)]
            if branchpointindices[i] > 0:
                colormap[branchpointarray[j,i]] = hexcolors[j] 
        
        plt.title('Branchpoint connectivity for the word: ' + '"' + branchpointlist[j] + '"', fontsize=20)
        nx.draw_circular(G, node_color=colormap, node_size=300, linewidths=0.5, font_size=12, font_weight='normal', with_labels=True)
        plt.savefig('networkx_branchpoint_' + j.__str__().zfill(3) +'.png')
        plt.close(fig)

    # Generate animated GIF

    fp_in = "networkx_branchpoint_*.png"
    fp_out = "networkx_branchpoints.gif"

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=1000, loop=0)
    
if plot_networkx_connections_dags == True:

    print('plotting_networkx_connections_dags ...')

    # Construct edgelist, labellist
    
    edgelist = [(i,i+1) for i in range(nwords-1)]
    labellist = [{i : wordlist[i]} for i in range(nwords)]

    for j in range(np.size(branchpointarray, axis=0)): # i.e. nbranchpoints        
        
        fig, ax = plt.subplots(figsize=(15,10))    
        G = nx.DiGraph()
        G.add_edges_from(edgelist)
        for node in G.nodes():
            G.nodes[node]['label'] = labellist[node]

        branchpointedges = []
        for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
            branchpointindices = branchpointarray[j,:]
            connections = branchpointindices[(branchpointindices != branchpointindices[i]) & (branchpointindices > 0)]
            for k in range(len(connections)):
                if branchpointindices[i] > 0:
                    branchpointedges.append([branchpointindices[i], connections[k]])
        G.add_edges_from(branchpointedges)

        # Colormap per branchpoint
    
        colormap = []
        for k in range(nwords):
            colormap.append('lightgrey')              
        for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
            branchpointindices = branchpointarray[j,:]
            connections = branchpointindices[(branchpointindices != branchpointindices[i]) & (branchpointindices > 0)]
            if branchpointindices[i] > 0:
                colormap[branchpointarray[j,i]] = hexcolors[j] 
        
        plt.title('Directed acyclic graph for the word: ' + '"' + branchpointlist[j] + '"', fontsize=20)
        pos = nx.spring_layout(G,iterations=200)       
        nx.draw_networkx(G, pos=pos, node_color=colormap, node_size=300, linewidths=0.5, font_size=12, font_weight='normal', with_labels=True)
        plt.savefig('networkx_dag_' + j.__str__().zfill(3) +'.png')
        plt.close(fig)

    # Generate animated GIF

    fp_in = "networkx_dag_*.png"
    fp_out = "networkx_dags.gif"

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=1000, loop=0)
            
if generate_adjacency:

    #--------------------------------------------------------------------------
    # CALCULATE ADJACENCY MATRIX FROM GRAPH
    #--------------------------------------------------------------------------

    print('computing_adjacency ...')


    G = nx.DiGraph()
#    G.add_edges_from(edgelist)
    for node in G.nodes():
        G.nodes[node]['label'] = labellist[node]
    for j in range(np.size(branchpointarray, axis=1)): # i.e. nbranchpoints        
        branchpointedges = []
        for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
            branchpointindices = branchpointarray[j,:]
            connections = branchpointindices[(branchpointindices != branchpointindices[i]) & (branchpointindices > 0)]
            for k in range(len(connections)):
                branchpointedges.append([branchpointindices[i], connections[k]])
        G.add_edges_from(branchpointedges)            

    A = nx.adjacency_matrix(G)
#    A.setdiag(A.diagonal()*2)

    df = pd.DataFrame(A.todense()) # convert sparse matrix to square array
    df.to_csv('adjacency_matrix.csv', sep=',', index=False, header=False, encoding='utf-8')

    # REPRODUCIBILITY TEST: plot from adjacency matrix
            
#    A = pd.read_csv('adjacency_matrix.csv', index_col=0)

#    fig, ax = plt.subplots(figsize=(15,10))
#    G = nx.from_numpy_matrix(np.array(A))
#    nx.draw_circular(G, node_color=branchpoint_colormap, node_size=500, linewidths=0.5, font_size=8, font_weight='normal', with_labels=True)    
#    plt.savefig('networkx_adjacency.png')

#    fig, ax = plt.subplots(figsize=(15,10))
#    G = nx.DiGraph(A.values)
#    nx.draw_networkx(G, node_color=branchpoint_colormap, node_size=500, linewidths=0.5, font_size=8, font_weight='bold', with_labels=True)
#    plt.savefig('networkx_adjacency2.png')

if generate_qubits:

    #--------------------------------------------------------------------------
    # QUBIT CONSTRUCTION
    #--------------------------------------------------------------------------
    print('generating_qubits ...')

if qubit_logic:

    #--------------------------------------------------------------------------
    # APPLY GATES TO BELL STATES
    #--------------------------------------------------------------------------
    print('applying logic gates ...')

if machine_learning:

    #--------------------------------------------------------------------------
    # FEATURE EXTRACTION
    #--------------------------------------------------------------------------
    print('extracting features ...')

if write_log:
            
    #--------------------------------------------------------------------------
    # LOG FILE
    #--------------------------------------------------------------------------
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
    text_file.write('BRANCHPOINTS (>1 connection): %s' % branchpointlist)
    text_file.write('\n\n')
    text_file.write('N(sentences)=%s' % nsentences)
    text_file.write('\n')
    text_file.write('N(lines)=%s' % nlines)
    text_file.write('\n')
    text_file.write('N(words)=%s' % nwords)
    text_file.write('\n')
    text_file.write('N(unique)=%s' % nunique)
    text_file.write('\n')
    text_file.write('N(branchpoints)=%s' % nbranchpoints)    
    text_file.write('\n')
    text_file.write('N(variants)=%s' % nvariants)    
    text_file.close()
    #--------------------------------------------------------------------------

print('** END')

     
