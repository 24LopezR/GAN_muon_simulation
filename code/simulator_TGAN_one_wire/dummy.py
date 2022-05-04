#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 08:50:29 2022

@author: ruben
"""
import numpy as np
from math import factorial

def combine(n=256):
    v = []
    for i in range(n):
        s = np.base_repr(i, 2)
        sum_i = np.sum(np.asarray(list(s), dtype=int))
        if sum_i>4:
            v.append(0)
        else:
            v.append(1)
        if i % 100000 == 0:
            print('Checked: '+str(i))
    return v

def combine_f(n=8):
    s = 0
    for i in range(4):
        s += factorial(n)/(factorial(n-i+1))
    return 2**n - s

v = combine(256)
v2 = combine_f(8)
print(np.sum(np.asarray(v)))
print(np.sum(np.asarray(v2)))