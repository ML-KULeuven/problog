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

import unittest

from problog.program import PrologString
from problog.engine import DefaultEngine
from problog.logic import Term, Constant
from problog.formula import LogicFormula

import glob, os, random

class TestEngine(unittest.TestCase):

    def setUp(self) :

        try :
            self.assertCollectionEqual = self.assertItemsEqual
        except AttributeError :
            self.assertCollectionEqual = self.assertCountEqual


    def test_nonground_query_ad(self) :
        """Non-ground call to annotated disjunction"""

        program = """
            0.1::p(a); 0.2::p(b).
            query(p(_)).
        """

        engine = DefaultEngine()
        db = engine.prepare( PrologString(program) )

        result=None
        for query in engine.query(db, Term('query', None)) :
            result = engine.ground(db, query[0], result, label='query')

        found = [ str(x) for x, y in result.queries() ]

        self.assertCollectionEqual( found, [ 'p(a)', 'p(b)' ])


    def test_compare(self) :
        """Comparison operator"""

        program = """
            morning(Hour) :- Hour >= 6, Hour =< 10.
        """

        engine = DefaultEngine()
        db = engine.prepare( PrologString(program) )

        self.assertEqual( list(map(list,engine.query(db, Term('morning', Constant(8)) ))), [[8]])

    def test_anonymous_variable(self) :
        """Anonymous variables are distinct"""

        program = """
            p(_,X,_) :- X = 3.

            q(1,2,3).
            q(1,2,4).
            q(2,3,5).
            r(Y) :- q(_,Y,_).

        """

        engine = DefaultEngine()
        db = engine.prepare( PrologString(program) )
        self.assertEqual( list(map(list,engine.query(db, Term('p', Constant(1), Constant(3), Constant(2) )))), [[Constant(1),Constant(3),Constant(2)]])

        self.assertEqual(list(map(list,engine.query(db, Term('r', None )))), [[2], [3]])

    def test_functors(self) :
        """Calls with functors"""

        program = """
            p(_,f(A,B),C) :- A=y, B=g(C).
            a(X,Y,Z) :- p(X,f(Y,Z),c).
        """
        pl = PrologString(program)

        r1 = DefaultEngine().query(pl, Term('a',Term('x'),None,Term('g',Term('c'))))
        r1 = [ list(map(str,sol)) for sol in r1  ]
        self.assertCollectionEqual( r1, [['x', 'y', 'g(c)']])

        r2 = DefaultEngine().query(pl, Term('a',Term('x'),None,Term('h',Term('c'))))
        self.assertCollectionEqual( r2, [])

        r3 = DefaultEngine().query(pl, Term('a',Term('x'),None,Term('g',Term('z'))))
        self.assertCollectionEqual( r3, [])


class TestEngineCycles(unittest.TestCase):

    def setUp(self) :
        self.edges = [(1,3),(1,5),(1,6),(1,8),(2,3),(2,5),(2,8)]
        self.edges += [(4,5),(4,7),(4,9),(4,11),(5,9),(6,10),(6,11),(6,13)]
        self.edges += [(7,11),(7,12),(8,9),(8,10),(8,11),(9,11),(10,11),(11,14)]
        self.edges += [ (y,x) for x,y in self.edges ]
        self.edges = list(sorted(self.edges))
        self.nodes = set([ x for x,y in self.edges ] + [ y for x,y in self.edges ])

        self.clause_v1 = [ 'active(X) :- edge(X,Y), active(Y).' ]
        self.clause_v2 = [ 'active(X) :- active(Y), edge(X,Y).' ]
        self.clauses = [ 'active(X) :- active_p(X).', 'query(active(11)).', 'path_from(X) :- path(X,_).' ]

        self.facts = [ '0.2::active_p(%s).' % e for e in self.nodes ] + [ '0.3::edge(%s,%s).' % e for e in self.edges ]

        self.program_v1 = self.facts + self.clauses + self.clause_v1
        self.program_v2 = self.facts + self.clauses + self.clause_v2

        self.maxDiff = None
        try :
            self.assertCollectionEqual = self.assertItemsEqual
        except AttributeError :
            self.assertCollectionEqual = self.assertCountEqual



    def test_cycle_goodcode(self) :
        N = 20
        program = self.program_v1[:]

        for i in range(0,N) :
            seed = str(random.random())[2:]
            random.seed(seed)
            random.shuffle(program)
            txt = '\n'.join(program)
            f = DefaultEngine(label_all=True).ground_all(PrologString(txt))
            paths = list(list_paths(f))

            edges = set()
            for p in paths :
                for i in range(0, len(p)-1) :
                    edges.add( ( int(p[i]), int(p[i+1]) ) )
            edges = list(sorted(edges))

            # if (edges != self.edges) :
            #     with open('cycle_error.pl', 'w') as f :
            #         print(txt, file=f)
            #     with open('cycle_error.dot', 'w') as f :
            #         print('digraph CycleError {', file=f)
            #         for edge in edges :
            #             print('%s -> %s;' % edge, file=f)
            #         print('}', file=f)

            self.assertCollectionEqual(self.edges, edges, msg='Test failed for random seed %s' % seed)

    def test_cycle_badcode(self) :
        N = 20
        program = self.program_v2[:]

        for i in range(0,N) :
            seed = str(random.random())[2:]
            random.seed(seed)
            random.shuffle(program)
            txt = '\n'.join(program)
            # try :
            f = DefaultEngine(label_all=True).ground_all(PrologString(txt))
            # except Exception :
            #     with open('cycle_error.pl', 'w') as f :
            #         print(txt, file=f)
            #     raise


            paths = list(list_paths(f))

            edges = set()
            for p in paths :
                for i in range(0, len(p)-1) :
                    edges.add( ( int(p[i]), int(p[i+1]) ) )
            edges = list(sorted(edges))

            # if (edges != self.edges) :
            #     with open('cycle_error.pl', 'w') as f :
            #         print(txt, file=f)
            #     with open('cycle_error.dot', 'w') as f :
            #         print('digraph CycleError {', file=f)
            #         for edge in edges :
            #             print('%s -> %s;' % edge, file=f)
            #         print('}', file=f)


            self.assertCollectionEqual(self.edges, edges, msg='Test failed for random seed %s' % seed)


def list_paths(source) :
    names = dict( (index, str(source.get_name(index))) for index, node, tp in source if index != 0 )

    for name, node in source.queries() :
        for path in _list_paths(source, node, []) :
            path = [ names.get(n) for n in path ]
            path = [ int(p[7:-1]) for p in path if p != None and p.startswith('active(')  ]
            yield path
        break

from itertools import product
def _list_paths( source, node_id, path ) :
    if node_id in path :
        yield [node_id]
    else :
        add = [node_id]
        node = source.get_node(node_id)
        nodetype = type(node).__name__
        if nodetype == 'conj' :
            c1, c2 = node.children
            paths1 = list( _list_paths(source,c1, path + add ) )
            paths2 = list( _list_paths(source,c2, path + add ) )
            for p1,p2 in product( paths1, paths2 ) :
                yield add + p1 + p2
        elif nodetype == 'disj' :
            for c in node.children :
                for p in _list_paths(source,c, path + add) :
                    yield add + p
        elif nodetype == 'atom' :
            yield add
        else :  # Don't support negation for now
            raise Exception('Unexpected node type')
