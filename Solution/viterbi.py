#!/usr/bin/env python3.6
import sys
import numpy as np
import pandas as pd
from math import log

prob_file = sys.argv[1]
words_file = sys.argv[2]

pos_tags = ["noun", "verb", "inf", "prep", "phi"]

prob_matrix = {}

transition_matrix = {}
emission_matrix = {}

word_dict = {}


with open(prob_file) as f:
    for x in f.readlines():
        arr = x.strip().split(' ')
        prob_matrix[(arr[0], arr[1])] = float(arr[2])

for x in prob_matrix:
    if x[0] in pos_tags and x[1] in pos_tags:
        transition_matrix[x] = prob_matrix[x]
    else:
        emission_matrix[x] = prob_matrix[x]

words = "bear fishes".split(' ')


def sequence(score, w):
    print(score)

def viterbi(words, pos_tags):

    c = 0
    for w in words:
        word_dict[c] = w
        w = c
        c += 1
    words = list(word_dict.keys())

    col_length = len(words)
    row_length = len(pos_tags)
    score = pd.DataFrame(np.zeros(col_length*row_length).reshape(row_length, col_length), index = pos_tags, columns = words)
    backpointers = score.copy()
    backpointers = backpointers[words].astype(str)
    # Initialization
    w = words[0]
    for t in pos_tags:
        pw = emission_matrix.get((word_dict[w], t))
        if pw is None:
            pw = 0.0001

        pt = transition_matrix.get((t, "phi"))
        if pt is None:
            pt = 0.0001

        score.at[t, w] = log(pw, 2) + log(pt, 2)
        backpointers.at[t, w] = str(0)

    # Iteration
    for i, w in enumerate(words):
        if i != 0:
            w_prev = words[i - 1]
            for j, t in enumerate(pos_tags):
                max_sum_dict = {}

                for k,a in enumerate(pos_tags):
                    pt = transition_matrix.get((t, a))
                    if pt is None:
                        pt = 0.0001
                    s = float(score.at[a, w_prev])
                    max_sum_dict[a] = (s + log(pt, 2))
                max_sum_k = max(max_sum_dict, key = max_sum_dict.get)
                max_sum = max_sum_dict[max_sum_k]

                # score_list = score.ix[:,w_prev].tolist()

                # prev_t = t
                # if j!=0:
                #     prev_t = pos_tags[j-1]

                # tag_list = [transition_matrix[x] for x in transition_matrix.keys() if x[1] == prev_t]

                # max_sum = [(x,y) for x in score_list for y in tag_list]
                # max_sum = [x[0] + log(x[1],2) for x in max_sum]
                # max_sum = max(max_sum)

                pw = emission_matrix.get((word_dict[w], t))
                if pw is None:
                    pw = 0.0001
                score.at[t, w] = log(pw, 2) + max_sum
                backpointers.at[t, w] = max_sum_k

    print(backpointers)
    # print(score)
    seq = [score[words[-1]].idxmax()]
    w_p = words[-1]
    words.reverse()
    for w in words[1:]:
        seq.append(backpointers.at[seq[-1], w_p])
        w_p = w
    seq.reverse()
    return seq


print(viterbi(words, pos_tags))
print(word_dict)
