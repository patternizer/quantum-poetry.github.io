#! /usr/bin/ python

# -*- coding: utf-8 -*-

#------------------------------------------------------------------------------
# PROGRAM: quantum_poetry_3d.py
#------------------------------------------------------------------------------
# Version 0.1
# 2 August, 2020
# Dr Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -----------------------------------------------------------------------------
# METHOD 1:
# -----------------------------------------------------------------------------

def explode(data):
    size = np.array(data.shape)*2
    data_e = np.zeros(size - 1, dtype=data.dtype)
    data_e[::2, ::2, ::2] = data
    return data_e

# initialise empty voxel cube (line, word, variant)
#n_voxels = np.zeros((14, 8, 94), dtype=bool)
n_voxels = np.zeros((8, 14, 8), dtype=bool)

n_voxels[0, :, 7] = True
n_voxels[-1, 0, :] = True
n_voxels[1, 0, 2] = True
n_voxels[2, 0, 1] = True

facecolors = np.where(n_voxels, '#ff619b', '#14d0f0')
edgecolors = np.where(n_voxels, '#d41243', '#0099f7')
filled = np.ones(n_voxels.shape)

# upscale the above voxel image, leaving gaps
filled_2 = explode(filled)
facecolors_2 = explode(facecolors)
edgecolors_2 = explode(edgecolors)

# Shrink the gaps
x, y, z = np.indices(np.array(filled_2.shape) + 1).astype(float) // 2
x[0::2, :, :] += 0.05
y[:, 0::2, :] += 0.05
z[:, :, 0::2] += 0.05
x[1::2, :, :] += 0.95
y[:, 1::2, :] += 0.95
z[:, :, 1::2] += 0.95

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(x, y, z, filled_2, facecolors=facecolors_2, edgecolors=edgecolors_2, alpha=0.25)
ax.set_xlabel('line')
ax.set_ylabel('word in line')
ax.set_zlabel('variant')
plt.savefig('3d_voxels1.png')


# -----------------------------------------------------------------------------
# METHOD 2:
# -----------------------------------------------------------------------------

# prepare coordinates
nlines = 8
#nvariants = 94
nvariants = 4
nmaxwords = 14
x, y, z = np.indices((nlines, nmaxwords, nvariants)) 

# draw separate colour segment cuboids for each line

prior = np.zeros((nlines, nmaxwords, nvariants), dtype=bool)
posterior = np.zeros((nlines, nmaxwords, nvariants), dtype=bool)
empty = np.zeros((nlines, nmaxwords, nvariants), dtype=bool)

for variant in range(nvariants):
    for line in range(nlines):
    
        # line cuboids
        lineprior = (x==line) & (y<5) & (z==variant)
        lineposterior = (x==line) & (y>=5) & (z==variant)  
        lineempty = (x==line) & (y==13) & (z==variant)
        
        # boolean append
        prior = prior | lineprior
        posterior = posterior | lineposterior
        empty = empty | lineempty

#prior = (x==6) & (y<5) & (z==variant)
#posterior = (x==6) & ((y==5) | (y==6)) & (z==variant)  
#empty = (x==6) & (y>7) & (z==variant)

# combine the objects into a single boolean array
voxels = prior | posterior | empty 

# set facecolors of voxel objects
fcolors = np.empty(voxels.shape, dtype=object)
fcolors[prior] = '#ff619b' # pink
fcolors[posterior] = '#14d0f0' # cyan
fcolors[empty] = '#d2e6f2' # lightgrey
#fcolors[empty] = '#ffffff' # white

# set edgecolors of voxel objects
ecolors = np.empty(voxels.shape, dtype=object)
ecolors[prior] = '#d41243' # red
ecolors[posterior] = '#0099f7' # blue
ecolors[empty] = '#bddaf0' # grey
#ecolors[empty] = '#ffffff' # white

# render 3D plot
fontsize=20
fig = plt.figure(figsize=(15,10))
ax = fig.gca(projection='3d')
ax.voxels(voxels, facecolors=fcolors, edgecolors=ecolors, alpha=0.25)
ax.set_xlabel('line in text', fontsize=fontsize)
ax.set_ylabel('word in anyon', fontsize=fontsize)
ax.set_zlabel('variant in topological map', fontsize=fontsize)
plt.savefig('3d_voxels.png')


