#! /usr/bin/env python
"""
Part of the ProbLog distribution.

Copyright 2015 KU Leuven, DTAI Research Group

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from __future__ import print_function

import sys
import time
import os

from problog.program import PrologFile
from problog.logic import Term
from problog.evaluator import SemiringSymbolic, Evaluator
from problog.engine import DefaultEngine
from problog.ddnnf_formula import DDNNF
from problog.cnf_formula import CNF
from problog.sdd_formula import SDD


class Timer(object) :

    def __init__(self, msg) :
        self.message = msg
        self.start_time = None

    def __enter__(self) :
        self.start_time = time.time()

    def __exit__(self, *args) :
        print ('%s: %.4fs' % (self.message, time.time()-self.start_time))


def main(filename, with_dot, knowledge) :

    dotprefix = None
    if with_dot :
        dotprefix = os.path.splitext(filename)[0] + '_'

    model = PrologFile(filename)

    engine = DefaultEngine(label_all=True)

    with Timer('parsing') :
        db = engine.prepare(model)

    print ('\n=== Database ===')
    print (db)

    print ('\n=== Queries ===')
    queries = engine.query(db, Term( 'query', None ))
    print ('Queries:', ', '.join([ str(q[0]) for q in queries ]))

    print ('\n=== Evidence ===')
    evidence = engine.query(db, Term( 'evidence', None, None ))
    print ('Evidence:', ', '.join([ '%s=%s' % ev for ev in evidence ]))

    print ('\n=== Ground Program ===')
    with Timer('ground') :
        gp = engine.ground_all(db)
    print (gp)

    if dotprefix != None :
        with open(dotprefix + 'gp.dot', 'w') as f :
            print ( gp.toDot(), file=f)

    print ('\n=== Acyclic Ground Program ===')
    with Timer('acyclic') :
        gp = gp.makeAcyclic()
    print (gp)

    if dotprefix != None :
        with open(dotprefix + 'agp.dot', 'w') as f :
            print ( gp.toDot(), file=f)

    if knowledge == 'sdd' :
        print ('\n=== SDD compilation ===')
        with Timer('compile') :
            nnf = SDD.createFrom(gp)

        if dotprefix != None :
            nnf.saveSDDToDot(dotprefix + 'sdd.dot')

    else :
        print ('\n=== Conversion to CNF ===')
        with Timer('convert to CNF') :
            cnf = CNF.createFrom(gp)

        print ('\n=== Compile to d-DNNF ===')
        with Timer('compile') :
            nnf = DDNNF.createFrom(cnf)

    if dotprefix != None :
        with open(dotprefix + 'nnf.dot', 'w') as f :
            print ( nnf.toDot(), file=f)

    print ('\n=== Evaluation result ===')
    with Timer('evaluate') :
        result = nnf.evaluate()

    for it in result.items() :
        print ('%s : %s' % (it))

if __name__ == '__main__' :

    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename')
    argparser.add_argument('--with-dot', action='store_true')
    argparser.add_argument('-k', '--knowledge', choices=['nnf','sdd'], default='sdd')
    args = argparser.parse_args()

    main(**vars(args))
