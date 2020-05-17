#!/usr/bin/python
# -*- coding: utf-8 -*-

#-----------------------------------------------------------------------
# PROGRAM: worldlines.py
#-----------------------------------------------------------------------
# Version 0.1
# 16 May, 2020
# Dr Michael Taylor
# https://patternizer.github.io
# patternizer AT gmail DOT com
#-----------------------------------------------------------------------


def extract_unique(input_file, output_file):
  z = []
  with open(input_file,'r') as fileIn, open(output_file,'w') as fileOut:
      for line in fileIn:
          for word in line.split():
              if word not in z:
                 z.append(word)
                 fileOut.write(word+'\n')
                 
input_file = ''
output_file = ''
extract_unique(input_file, output_file)

f = open(output_file, 'r')
wordlist = f.read()
unique_words = set(wordlist)

for word in unique_words:
#	

	 