#!/usr/bin/env python3
# encoding: utf-8
"""
coin.py

Created by Wannes Meert on 17-11-2016.
Copyright (c) 2016 KU Leuven. All rights reserved.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import py2problog as p2p


def make_coin(weight):
    return lambda: 'h' if p2p.ProbFact(weight, 'pf_'+str(weight))() else 't'
def bend(coin):
    return lambda: make_coin(0.7) if coin() == 'h' else make_coin(0.1)

fair_coin = make_coin(0.5)
bent_coin = bend(fair_coin)

def run():
    return bent_coin()()


#########################

p2p.query(run)

with open('coin.dot', 'w') as ofile:
    print(p2p.dot(), file=ofile)



