#!/usr/bin/env python3
# encoding: utf-8
"""
example.py

Created by Wannes Meert on 17-11-2016.
Copyright (c) 2016 KU Leuven. All rights reserved.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import subprocess as sp
import py2problog as p2p
import logging
import sys

p2p.logger.setLevel(logging.DEBUG)
p2p.logger.addHandler(logging.StreamHandler(sys.stdout))
p2p.settings.use_trace = False


## Probabilistic Program

x = p2p.ProbFact(0.25, 'x')
y = p2p.ProbFact(0.75, 'y')
z = p2p.ProbFact(0.85, 'z')
a = p2p.AD([(0.25,'a'),(0.35,'b'),(0.40,'c')], 'a')


# def rule0(v):
    # if y() and z():
        # return 5,4
    # else:
        # return 3,2

# def rule1():
    # if x():
        # return True
    # else:
        # if rule0(4) > (4,1):
            # return True
        # else:
            # return False

# def rule2(r):
    # for i in range(r):
        # yield i*2

@p2p.probabilistic
def rule1():
    if x():
        return True
    else:
        return rule2()

@p2p.probabilistic
def rule2():
    if y() and z():
        return True
    elif a() == 'b' and det_func():
        return True
    else:
        return False

def det_func2(s):
    c = s
    d = c + 3
    return True

@p2p.deterministic
def det_func():
    a = 1
    b = a + 2
    return det_func2(0.4)


## Run inference

p2p.settings.use_trace = True
p2p.settings.nb_samples=10
p2p.query(rule1)
# p2p.query(rule2)
# p2p.query_es(rule1)

with open('example.dot', 'w') as ofile:
    print(p2p.dot(), file=ofile)
with open('example.pl', 'w') as ofile:
    print(p2p.problog(), file=ofile)


cmd = ["problog", "example.pl"]
print("\n$ {}".format(" ".join(cmd)))
sp.run(cmd)

print("\nCount samples:")
cnt = p2p.count_samples(rule1)
print(cnt)
print(", ".join(["{}={}".format(k, v/p2p.settings.nb_samples) for k,v in cnt.items()]))

# p2p.hist_samples(rule1)

