#!/usr/bin/env python3
# encoding: utf-8
"""
higherorder.py

Created by Wannes Meert on 17-11-2016.
Copyright (c) 2016 KU Leuven. All rights reserved.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import subprocess as sp
from functools import reduce
import py2problog as p2p


## Probabilistic Program

coins = [p2p.ProbFact(0.8, 'coin_{}'.format(i)) for i in range(5)]

def data():
    return sum(map(lambda x: 1 if x() else 0, coins))

# Better with caching (failed)
def data2():
    return reduce(lambda r,c: r+1 if c() else r, coins, 0) # this include the c element in the arguments and makes it impossible to cache
    # lambdas are in general not a good way or representing data. We need to
    # cache on all local variables because we don't know which ones are
    # relevant in the scope.  For defs we assume the arguments are sufficient
    # and we ignore scoped variables (we probably can also figure out which
    # scoped variables are relevant using the inspect module).

# Better with caching
def cumsum(d=0):
    if d >= len(coins):
        return 0
    s = 1 if coins[d]() else 0
    s += cumsum(d+1)
    return s

def data3():
    return cumsum()


## Run inference

p2p.settings.nb_samples = 100
# p2p.query(data)
# p2p.query_sampling([(data3,None), (data3,(4,))])
p2p.query(data3)
# p2p.query(data3,(4,))
# p2p.query_es(data)

with open('higherorder.dot', 'w') as ofile:
    print(p2p.dot(), file=ofile)
with open('higherorder.pl', 'w') as ofile:
    print(p2p.problog(), file=ofile)


cmd = ["problog", "higherorder.pl"]
print("\n$ {}".format(" ".join(cmd)))
sp.run(cmd)


print("\nCount samples:")
cnt = p2p.count_samples(data)
print(cnt)
print(", ".join(["{}={}".format(k, v/p2p.settings.nb_samples) for k,v in cnt.items()]))


