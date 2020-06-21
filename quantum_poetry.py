#!/usr/bin/python
# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: worldlines.py
#------------------------------------------------------------------------------
# Version 0.6
# 20 June, 2020
# Dr Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# SETTINGS
write_log = True
plot_branchpoint_table = True
generate_anyons = True
generate_qubits = True
plot_networkx_connections = True
plot_networkx_connections_knots = False
plot_networkx_connections_braids = False
plot_networkx_non_circular = False
compute_erdos_parameter = True
compute_erdos_equivalence = False
generate_adjacency = False
qubit_logic = False
machine_learning = False
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# IMPORT PYTHON LIBRARIES
#------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import scipy as sp
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
import plotly.express as px
# NLP Libraries
# ML Libraries
# App Libraries

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
#------------------------------------------------------------------------------

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
        
nsentences = len(sentencelist)    # --> 10
nlines = len(linelist)            # --> 26
nwords = len(wordlist)            # --> 222
nunique = len(uniquewordlist)     # --> 134
nknots = len(knotlist)            # --> 30

# Branchpoint index array

maxbranches = wordfreq[0][1]
branchpointarray = np.zeros((nknots, maxbranches), dtype='int')
for k in range(len(knotlist)):  
    index = []
    for i, j in enumerate(wordlist):
        if j == knotlist[k]:
            index.append(i)            
    branchpointarray[k,0:len(index)] = index
 
df = pd.DataFrame(branchpointarray)
df.to_csv('branchpointarray.csv', sep=',', index=False, header=False, encoding='utf-8')
    
#------------------------------------------------------------------------------
# CONSTRUCT COLORMAP
#------------------------------------------------------------------------------

freq = [ wordfreq[i][1] for i in range(len(wordfreq)) ]
nlabels = nknots
cmap = px.colors.diverging.Spectral
cmap_idx = np.linspace(0,len(cmap)-1, nlabels, dtype=int)
colors = [cmap[i] for i in cmap_idx]
hexcolors = [ rgb2hex(colors[i]) for i in range(len(colors)) ]
    
knot_colormap = []
for k in range(nwords):    
    knot_colormap.append('lightgrey')              
for j in range(np.size(branchpointarray, axis=0)): # i.e. nknots            
    for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
        knot_colormap[branchpointarray[j,i]] = hexcolors[j] 
#------------------------------------------------------------------------------
                                  
if plot_branchpoint_table:
    
    #--------------------------------------------------------------------------
    # BRANCHPOINT PLOT
    #--------------------------------------------------------------------------

    print('plotting_branchpoint_table ...')

    # Reconstruction test: colour branchpoints with knot connectivity
    fig, ax = plt.subplots(figsize=(15,10))
    plt.plot(np.arange(0,len(wordlist)), np.zeros(len(wordlist)))
    for k in range(len(knotlist)):  
        plt.plot(np.arange(0,len(wordlist)), np.ones(len(wordlist))*k, color='black')
        a = branchpointarray[k,:]
        vals = a[a>0]
        plt.scatter(vals, np.ones(len(vals))*k, label=knotlist[k], s=100, color=hexcolors[k])
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('word n in text', fontsize=20)
    plt.ylabel('knot k in text (>1 connection)', fontsize=20)
    plt.title('Branch Analysis Plot', fontsize=20)
    plt.gca().invert_yaxis()    
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.savefig('branchplot.png')
    plt.close(fig)

if plot_networkx_connections:
    
    #--------------------------------------------------------------------------
    # NETWORK CONNECTIVITY PLOT - G
    #--------------------------------------------------------------------------
        
    print('plotting_networkx_connections ...')
    
    # Construct edgelist, labellist
    
    edgelist = [(i,i+1) for i in range(nwords-1)]
    labellist = [{i : wordlist[i]} for i in range(nwords)]

    df = pd.DataFrame()
    
    # Plot wordfreq colour-coded networkx graph of connectivity
    
    fig, ax = plt.subplots(figsize=(15,10))    
    G = nx.Graph()
    G.add_edges_from(edgelist)
#    G.add_nodes_from(wordlist)
#    G.add_nodes_from(labellist)
#    [ G.add_node(wordlist[i]) for i in range(len(wordlist)-1) ]        
    for node in G.nodes():
        G.nodes[node]['label'] = labellist[node]

    edge_colormap = []
    for k in range(nwords-1):
        edge_colormap.append('lightgrey')              
        
    for j in range(np.size(branchpointarray, axis=0)): # i.e. nknots        
        knotedges = []
        for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
            knotindices = branchpointarray[j,:]
            connections = knotindices[(knotindices != knotindices[i]) & (knotindices > 0)]
            for k in range(len(connections)):
                if knotindices[i] > 0:
                    knotedges.append([knotindices[i], connections[k]])
        G.add_edges_from(knotedges)        
        for l in range(int(len(knotedges)/2)): # NB 2-driectional edges
            edge_colormap.append(hexcolors[j])
#    edge_colormap = [G[u][v]['color'] for u,v in edges]        
#    nx.draw_circular(G, node_color=knot_colormap, edge_color=edge_colormap, node_size=500, linewidths=0.5, font_size=8, font_weight='normal', with_labels=True)
    nx.draw_circular(G, node_color=knot_colormap, node_size=500, linewidths=0.5, font_size=8, font_weight='normal', with_labels=True)
    plt.title('Networkx (circularly connected): N(edges)=' + "{0:.0f}".format(len(G.edges)))
    plt.savefig('networkx.png')
    plt.close(fig)

    nedges = len(G.edges)
    
if plot_networkx_connections_knots:
    
    #--------------------------------------------------------------------------
    # NETWORK CONNECTIVITY PLOT PER KNOT 
    #--------------------------------------------------------------------------

    print('plotting_networkx_connections_knots ...')

    # Construct edgelist, labellist
    
    edgelist = [(i,i+1) for i in range(nwords-1)]
    labellist = [{i : wordlist[i]} for i in range(nwords)]

    for j in range(np.size(branchpointarray, axis=0)): # i.e. nknots        
        
    # Plot wordfreq colour-coded networkx graph of connectivity
     
        fig, ax = plt.subplots(figsize=(15,10))    
        G = nx.DiGraph()
        G.add_edges_from(edgelist)
        for node in G.nodes():
            G.nodes[node]['label'] = labellist[node]

        knotedges = []
        for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
            knotindices = branchpointarray[j,:]
            connections = knotindices[(knotindices != knotindices[i]) & (knotindices > 0)]
            for k in range(len(connections)):
                if knotindices[i] > 0:
                    knotedges.append([knotindices[i], connections[k]])
        G.add_edges_from(knotedges)

        # Colormap per knot
    
        colormap = []
        for k in range(nwords):
            colormap.append('lightgrey')              
        for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
            knotindices = branchpointarray[j,:]
            connections = knotindices[(knotindices != knotindices[i]) & (knotindices > 0)]
            if knotindices[i] > 0:
                colormap[branchpointarray[j,i]] = hexcolors[j] 
        
        plt.title('Anyon connectivity for the word: ' + '"' + knotlist[j] + '"', fontsize=20)
        nx.draw_circular(G, node_color=colormap, node_size=500, linewidths=0.5, font_size=8, font_weight='normal', with_labels=True)
        plt.savefig('networkx_knot_' + j.__str__() +'.png')
        plt.close(fig)
        
if plot_networkx_connections_braids:
    
    #--------------------------------------------------------------------------
    # BRAIDS PER KNOT
    #--------------------------------------------------------------------------

    print('plotting_networkx_connections_braids ...')

    # Construct edgelist, labellist
    
    edgelist = [(i,i+1) for i in range(nwords-1)]
    labellist = [{i : wordlist[i]} for i in range(nwords)]

    for j in range(np.size(branchpointarray, axis=0)): # i.e. nknots        
        
        fig, ax = plt.subplots(figsize=(15,10))    
        G = nx.DiGraph()
        G.add_edges_from(edgelist)
        for node in G.nodes():
            G.nodes[node]['label'] = labellist[node]

        knotedges = []
        for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
            knotindices = branchpointarray[j,:]
            connections = knotindices[(knotindices != knotindices[i]) & (knotindices > 0)]
            for k in range(len(connections)):
                if knotindices[i] > 0:
                    knotedges.append([knotindices[i], connections[k]])
        G.add_edges_from(knotedges)

#        paths = nx.all_simple_paths(G, source=0, target=(nwords-1))
#        paths = nx.shortest_path(G, source=0, target=(nwords-1))
#        paths = nx.single_source_shortest_path(G, source=0, target=(nwords-1)) 
#        paths = nx.all_shortest_paths(G, source=0, target=(nwords-1), weight=None, method='dijkstra')
#        paths = nx.all_shortest_paths(G, source=0, target=(nwords-1), weight=None, method='bellman-ford')

        # Colormap per knot
    
        colormap = []
        for k in range(nwords):
            colormap.append('lightgrey')              
        for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
            knotindices = branchpointarray[j,:]
            connections = knotindices[(knotindices != knotindices[i]) & (knotindices > 0)]
            if knotindices[i] > 0:
                colormap[branchpointarray[j,i]] = hexcolors[j] 
        
        plt.title('Anyon braids for the word: ' + '"' + knotlist[j] + '"', fontsize=20)
        pos = nx.spring_layout(G,iterations=200)       
        nx.draw_networkx(G, pos=pos, node_color=colormap, node_size=500, linewidths=0.5, font_size=8, font_weight='normal', with_labels=True)
        plt.savefig('networkx_braid_' + j.__str__() +'_.png')
        plt.close(fig)

if plot_networkx_non_circular:

    #--------------------------------------------------------------------------
    # PLOT NETWORK KNOT CONNECTIVITY
    #--------------------------------------------------------------------------

    print('plotting_networkx_non_circular ...')

    # Generate non-circular form of the networkx graph

    N = nx.Graph()
    N.add_edges_from(edgelist)
    for j in range(np.size(branchpointarray, axis=0)): # i.e. nknots        
        knotedges = []
        for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
            knotindices = branchpointarray[j,:]
            connections = knotindices[(knotindices != knotindices[i]) & (knotindices > 0)]
            for k in range(len(connections)):
                if knotindices[i] > 0:
                    knotedges.append([knotindices[i], connections[k]])
        N.add_edges_from(knotedges)        
    N.remove_edges_from(edgelist)
    N_degrees = [degree for node,degree in dict(N.degree()).items()] # degree of nodes
    notknots = [ node for node,degree in dict(N.degree()).items() if degree == 0 ] # each node in circular graph has 2 neighbours at start

    fig, ax = plt.subplots(figsize=(15,10))
    nx.draw_circular(N, node_color='lightgrey', node_size=500, linewidths=0.5, font_size=8, font_weight='normal', with_labels=True)
    plt.title('Networkx (non-circularly connected): N(edges)=' + "{0:.0f}".format(len(N.edges)))
    plt.savefig('networkx_non_circular.png')
        
if compute_erdos_parameter:

    #--------------------------------------------------------------------------
    # ERDOS-RENYI ESTIMATE
    #--------------------------------------------------------------------------

    print('computing_erdos_parameter ...')

    import random
    for connectivity in np.linspace(0,1,1000001):
        random.seed(42)
        E = nx.erdos_renyi_graph(nwords, connectivity)
        erdosedges = len(E.edges)
        if erdosedges == (nedges-len(edgelist)):            
            print("{0:.6f}".format(connectivity))
            print("{0:.6f}".format(erdosedges))
            break
    nerdosedges = len(E.edges)

    fig, ax = plt.subplots(figsize=(15,10))
    nx.draw_circular(E, node_color='lightgrey', node_size=500, linewidths=0.5, font_size=8, font_weight='normal', with_labels=True)
    plt.title('Erdős-Rényi Model: p=' + "{0:.6f}".format(connectivity) + ', N(edges)=' + "{0:.0f}".format(nerdosedges))
    plt.savefig('networkx_erdos.png')
    plt.close(fig)

if compute_erdos_equivalence:

    #--------------------------------------------------------------------------
    # ERDOS-RENYI EQUIVALENCE
    #--------------------------------------------------------------------------

    print('computing_erdos_equivalence ...')

    # Compare Erdos-Renyi graph edges in reduced networks (branchpoint network)

    N.remove_nodes_from(notknots)
    mapping = { np.array(N.nodes)[i]:i for i in range(len(N.nodes)) }
    H = nx.relabel_nodes(N,mapping)
            
#    Ecopy = E.copy()
#    Ecopy.remove_edge(a,b)
#    Ecopy.add_edge(c,d)
    
    maxdiff = len(H.edges)
    for i in range(100001):
        fig, ax = plt.subplots(figsize=(15,10))
        E = nx.erdos_renyi_graph(len(H.nodes), connectivity)
        diff = H.edges - E.edges        
        if len(diff) < maxdiff:
            maxdiff = len(diff)
            common = H.edges - diff            
            nx.draw_circular(E, node_color='lightgrey', node_size=500, linewidths=0.5, font_size=8, font_weight='normal', with_labels=True)
            plt.title('Erdős-Rényi Model (equivalent): N(common edges)=' + "{0:.0f}".format(len(N.edges)-len(diff)))
            plt.savefig('networkx_erdos' + '_' + "{0:.0f}".format(i) + '.png')
        
if generate_adjacency:

    #--------------------------------------------------------------------------
    # CALCULATE ADJACENCY MATRIX FROM GRAPH
    #--------------------------------------------------------------------------

    print('generating_adjacency ...')


    G = nx.DiGraph()
#    G.add_edges_from(edgelist)
    for node in G.nodes():
        G.nodes[node]['label'] = labellist[node]
    for j in range(np.size(branchpointarray, axis=1)): # i.e. nknots        
        knotedges = []
        for i in range(np.size(branchpointarray, axis=1)): # i.e. maxfreq
            knotindices = branchpointarray[j,:]
            connections = knotindices[(knotindices != knotindices[i]) & (knotindices > 0)]
            for k in range(len(connections)):
                knotedges.append([knotindices[i], connections[k]])
        G.add_edges_from(knotedges)            

    A = nx.adjacency_matrix(G)
#    A.setdiag(A.diagonal()*2)

    df = pd.DataFrame(A.todense()) # convert sparse matrix to square array
    df.to_csv('adjacency_matrix.csv', sep=',', index=False, header=False, encoding='utf-8')

    # REPRODUCIBILITY TEST: plot from adjacency matrix
            
#    A = pd.read_csv('adjacency_matrix.csv', index_col=0)

#    fig, ax = plt.subplots(figsize=(15,10))
#    G = nx.from_numpy_matrix(np.array(A))
#    nx.draw_circular(G, node_color=knot_colormap, node_size=500, linewidths=0.5, font_size=8, font_weight='normal', with_labels=True)    
#    plt.savefig('networkx_adjacency.png')

#    fig, ax = plt.subplots(figsize=(15,10))
#    G = nx.DiGraph(A.values)
#    nx.draw_networkx(G, node_color=knot_colormap, node_size=500, linewidths=0.5, font_size=8, font_weight='bold', with_labels=True)
#    plt.savefig('networkx_adjacency2.png')

if generate_anyons:

    #--------------------------------------------------------------------------
    # ANYON CONSTRUCTION: BRAIDING
    #--------------------------------------------------------------------------
    print('generating_anyons ...')

    # Compute start and end word indices for each line of the poem

    lineindices = []    
    wordcount = 0
    for i in range(len(linelist)):
        linelen = len(linelist[i].split())
        lineindices.append([i,wordcount,wordcount+linelen-1])
        wordcount += linelen
                    
    # For each line find word indices to and from knot

    branchlinearray = []        
    for j in range(np.size(branchpointarray, axis=0)): # i.e. nknots        
        knotindices = branchpointarray[j,:][branchpointarray[j,:]>0]
        for l in range(len(linelist)):                     
            for k in range(len(knotindices)):
                if knotindices[k] in np.arange(lineindices[l][1],lineindices[l][2]+1):                   
                    branchlinearray.append([l,j,lineindices[l][1],knotindices[k],lineindices[l][2]])
    
    # Filter out multiple knot in single line only occurences

    a = np.array(branchlinearray)
    mask = []
    for i in range(len(branchlinearray)-2):
        if (a[i,0] == a[i+1,0]) & (a[i,1] == a[i+1,1]) & (a[i+2,1]!=a[i,1]):
            mask.append(i)
            mask.append(i+1)
    for i in range(len(mask)):
        a = np.delete(a,mask[i]-i,0)        
    asorted = a[a[:,0].argsort()]

    # line, knot, wordstart, wordknot, wordend 

    # Anyons
    
    anyonarray = []
    for i in range(len(linelist)):
        b = asorted[asorted[:,0]==i] 
        for j in range(len(b)):    
            anyon_pre = wordlist[b[j,2]:b[j,3]+1]
            c = asorted[(asorted[:,1]==b[j,1]) & (asorted[:,0]!=b[j,0])]             
            for k in range(len(c)):
                anyon_post = wordlist[c[k,3]+1:c[k,4]+1]
                anyon = anyon_pre + anyon_post
                anyonarray.append([i,c[k,0],knotlist[b[j,1]],anyon])
    df = pd.DataFrame(anyonarray)
    df.to_csv('anyonarray.csv', sep=',', index=False, header=False, encoding='utf-8')
                                                      
if generate_qubits:

    #--------------------------------------------------------------------------
    # QUBIT CONSTRUCTION
    #--------------------------------------------------------------------------
    print('generating_qubits ...')

    # form sets of qubits one for each variation of the poem

    poem = []
    lineidx = []    
    lines = np.arange(len(linelist))    
    
    a = df[df[0]==lines[0]]
    linestart = a[0].values[0]
    lineend = a[1].values[0]
    lineidx.append(linestart)    
    lineidx.append(lineend)
    knot = a[2].values[0]
    poem.append(df[(df[0]==linestart)&(df[1]==lineend)&(df[2]==knot)][3].values[0])
    poem.append(df[(df[0]==lineend)&(df[1]==linestart)&(df[2]==knot)][3].values[0])
    lines = np.setdiff1d(lines,lineidx)    
    
    while len(lines)>0:
        
        a = df[df[0]==lines[0]]
        linestart = a[0].values[0]
        lineend = np.setdiff1d( np.unique(a[1].values), lineidx )[0]    
        lineidx.append(linestart)    
        lineidx.append(lineend)
        knot = a[2].values[lineend]
        poem.append(df[(df[0]==linestart)&(df[1]==lineend)&(df[2]==knot)][3].values[0])
        poem.append(df[(df[0]==lineend)&(df[1]==linestart)&(df[2]==knot)][3].values[0])
        lines = np.setdiff1d(lines,lineidx)    

    df = pd.DataFrame(poem)
    df.to_csv('poem1.csv', sep=',', index=False, header=False, encoding='utf-8')

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
    #--------------------------------------------------------------------------

print('** END')

     
