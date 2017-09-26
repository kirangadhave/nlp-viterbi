#!/usr/bin/env python3.6
import sys

prob_file = sys.argv[1]
sent_file = sys.argv[2]

pos_tags =  ["noun", "verb", "inf", "prep"]

prob_matrix = {}

transition_matrix = {}
emission_matrix = {}

with open(prob_file) as f:
    for x in f.readlines():
        arr = x.strip().split(' ')
        prob_matrix[(arr[0], arr[1])] = float(arr[2])

for x in prob_matrix:
    if x[0] in pos_tags and x[1]  in pos_tags:
        transition_matrix[x] = prob_matrix[x]
    else:
        emission_matrix[x] = prob_matrix[x]
