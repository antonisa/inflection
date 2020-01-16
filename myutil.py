import numpy as np
import codecs
import re
from collections import defaultdict
from random import random, choice

def edit_distance(str1, str2):
    '''Simple Levenshtein implementation for evalm.'''
    table = np.zeros([len(str2) + 1, len(str1) + 1])
    for i in range(1, len(str2) + 1):
        table[i][0] = table[i - 1][0] + 1
    for j in range(1, len(str1) + 1):
        table[0][j] = table[0][j - 1] + 1
    for i in range(1, len(str2) + 1):
        for j in range(1, len(str1) + 1):
            if str1[j - 1] == str2[i - 1]:
                dg = 0
            else:
                dg = 1
            table[i][j] = min(table[i - 1][j] + 1, table[i][j - 1] + 1,
                              table[i - 1][j - 1] + dg)
    return int(table[len(str2)][len(str1)])

def read_data(filename):
    with codecs.open(filename, 'r', 'utf-8') as inp:
        lines = inp.readlines()
    inputs = []
    outputs = []
    tags = []
    for l in lines:
        l = l.strip().split('\t')
        if l:
            inputs.append(list(l[0]))
            outputs.append(list(l[1]))
            tags.append(re.split('\W+', l[2]))
    return inputs, outputs, tags

def read_test_data(filename):
    with codecs.open(filename, 'r', 'utf-8') as inp:
        lines = inp.readlines()
    inputs = []
    tags = []
    for l in lines:
        l = l.strip().split('\t')
        if l:
            inputs.append(list(l[0]))
            tags.append(re.split('\W+', l[1]))
    return inputs, tags

def read_bpe_data(filename):
    with codecs.open(filename, 'r', 'utf-8') as inp:
        lines = inp.readlines()
    inputs = []
    outputs = []
    tags = []
    for l in lines:
        l = l.strip().split('\t')
        if l:
            inputs.append(l[0].strip().split())
            outputs.append(l[1].strip().split())
            tags.append(re.split('\W+', l[2]))
    return inputs, outputs, tags

def write_vocab(l,filename):
    with open(filename, 'w') as outp:
        outp.write('\t'.join(l))

def read_vocab(filename):
    with open(filename) as inp:
        lines = inp.readlines()
    return lines[0].strip().split('\t')


def argmax(arr, k):
    k = min(k, arr.size)
    # get k best indices
    indices = np.argpartition(arr, -k)[-k:]
    # sorted
    indices = indices[np.argsort(arr[indices])]
    # flip so first has largest value
    return np.flip(indices, 0)



