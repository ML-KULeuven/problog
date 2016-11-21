#!/usr/bin/env python3
# encoding: utf-8
"""
probgraph.py

Created by Wannes Meert on 17-11-2016.
Copyright (c) 2016 KU Leuven. All rights reserved.
"""

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
import py2problog as p2p


## Probabilistic Program

edges = {
    'a': [(p2p.ProbFact(0.9,'a->b'),'b'), (p2p.ProbFact(0.3,'a->c'),'c')],
    'b': [(p2p.ProbFact(0.4,'b->c'),'c')],
    'c': [(p2p.ProbFact(0.5,'c->d'),'d')]
}

# def neighbors(f):
    # for pf,t in edges[f]:
        # if pf():
            # yield t

def neighbors(f):
    if not f in edges:
        return []
    return [t for pf,t in edges[f] if pf()]

def path(start, end, visited=()):
    if start == end: return True
    if start in visited: return False
    # return any((path(nxt,end,visited+(start,)) for nxt in neighbors(start)))
    return any([path(nxt,end,visited+(start,)) for nxt in neighbors(start)])


## Run inference

# p2p.settings.nb_samples = 1
p2p.query(path, args=('a','d'))

# Print call graph to Graphviz dot
with open('probgraph.dot', 'w') as ofile:
    print(p2p.dot(), file=ofile)

# Print call graph to problog and run inference
cmd = ["problog", "example.pl"]
print("\n$ {}".format(" ".join(cmd)))
sp.run(cmd)

# Perform inference using (naive) sampling
print("\nCount samples:")
cnt = p2p.count_samples(rule1)
print(cnt)
print(", ".join(["{}={}".format(k, v/p2p.settings.nb_samples) for k,v in cnt.items()]))

