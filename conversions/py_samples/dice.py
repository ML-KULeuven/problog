#!/usr/bin/env python3
# encoding: utf-8
"""
dice.py

Created by Wannes Meert on 17-11-2016.
Copyright (c) 2016 KU Leuven. All rights reserved.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import py2problog as p2p

## Probabilistic Program

def dice(i):
    # If prbabilistic facts are created while executing, they need a name
    return p2p.AD([(1/6,i) for i in range(1,7)], 'dice_{}'.format(i))


def dice_sum():
    return sum([dice(i)() for i in range(2)])


# dices = [p2p.AD([(1/6,i) for i in range(1,7)], 'dice_{}'.format(i)) for i in range(2)]

# def dice_sum():
    # return sum([dices[i]() for i in range(2)])


## Run inference

p2p.query(dice_sum)

with open('dice.dot', 'w') as ofile:
    print(p2p.dot(), file=ofile)




