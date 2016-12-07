#!/usr/bin/env python3
# encoding: utf-8
"""
geometric.py

Created by Wannes Meert on 17-11-2016.
Copyright (c) 2016 KU Leuven. All rights reserved.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import py2problog as p2p


## Probabilistic Program

def geometric(p, d=0):
    if p2p.ProbFact(p, 'stop'+str(d))():
        return 0
    else:
        return 1 + geometric(p, d+1)


## Run inference

p2p.settings.nb_samples = 100
p2p.query(geometric, args=(0.6,))

with open('geometric.dot', 'w') as ofile:
    print(p2p.dot(), file=ofile)





