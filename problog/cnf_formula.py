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

from .formula import LogicDAG

from .core import ProbLogObject, transform, deprecated, deprecated_function
from .util import Timer

from .evaluator import SemiringLogProbability

import warnings
import tempfile


# class CNF(ProbLogObject) :
#     """A logic formula in Conjunctive Normal Form.
#
#     This class does not derive from LogicFormula. (Although it could.)
#
#     """
#
#     def __init__(self, **kwdargs):
#         self.__atom_count = 0
#         self.__lines = []
#         self.__constraints = []
#
#         self.__names = []
#         self.__weights = []
#
#     def addAtom(self, *args) :
#         self.__atom_count += 1
#
#     def addAnd(self, content) :
#         raise TypeError('This data structure does not support conjunctions.')
#
#     def addOr(self, content) :
#         self.__lines.append( ' '.join(map(str, content)) + ' 0' )
#
#     def addNot(self, content) :
#         return -content
#
#     def addConstraint(self, constraint) :
#         self.__constraints.append(constraint)
#         for l in constraint.encodeCNF() :
#             self.__lines.append(' '.join(map(str,l)) + ' 0')
#
#     def constraints(self) :
#         return self.__constraints
#
#     def getNamesWithLabel(self) :
#         return self.__names
#
#     def setNamesWithLabel(self, names) :
#         self.__names = names
#
#     def getWeights(self) :
#         return self.__weights
#
#     def setWeights(self, weights) :
#         self.__weights = weights
#
#     def ready(self) :
#         pass
#
#     def to_dimacs(self):
#         return 'p cnf %s %s\n' % (self.__atom_count, len(self.__lines)) + '\n'.join( self.__lines )
#
#     toDimacs = deprecated_function('toDimacs', to_dimacs)
#
#     def getAtomCount(self) :
#         return self.__atom_count
#
#     def isTrivial(self) :
#         return len(self.__lines) == 0


# class CNFFormula(LogicDAG):
#     """A CNF stored in memory."""
#
#     def __init__(self, **kwdargs):
#         LogicDAG.__init__(auto_compact=False, **kwdargs)
#
#     def __iter__(self) :
#         for n in LogicDAG.__iter__(self) :
#             yield n
#         yield self._create_conj( tuple(range(self.getAtomCount()+1, len(self)+1) ) )
#
#     def addAnd(self, content) :
#         raise TypeError('This data structure does not support conjunctions.')
#
# class CNFFile(CNF) :
#     """A CNF stored in a file."""
#
#     # TODO add read functionality???
#
#     def __init__(self, filename=None, readonly=True, **kwdargs):
#         self.filename = filename
#         self.readonly = readonly
#
#         if filename is None :
#             self.filename = tempfile.mkstemp('.cnf')[1]
#             self.readonly = False
#         else :
#             self.filename = filename
#             self.readonly = readonly
#
#     def ready(self) :
#         if self.readonly :
#             raise TypeError('This data structure is read only.')
#         with open(self.filename, 'w') as f :
#             f.write('p cnf %s %s\n' % (self.__atom_count, len(self.__lines)))
#             f.write('\n'.join(self.__lines))

# @transform(LogicDAG, CNF)
# def clarks_completion(source, destination, **kwdargs):
#     with Timer('Clark\'s completion'):
#         # Every node in original gets a literal
#         num_atoms = len(source)
#
#         # Add atoms
#         for i in range(0, num_atoms) :
#             destination.addAtom( (i+1), True, (i+1) )
#
#         # Complete other nodes
#         for index, node, nodetype in source :
#             if nodetype == 'conj' :
#                 destination.addOr( (index,) + tuple( map( lambda x : destination.addNot(x), node.children ) ) )
#                 for x in node.children  :
#                     destination.addOr( (destination.addNot(index), x) )
#             elif nodetype == 'disj' :
#                 destination.addOr( (destination.addNot(index),) + tuple( node.children ) )
#                 for x in node.children  :
#                     destination.addOr( (index, destination.addNot(x)) )
#             elif nodetype == 'atom' :
#                 pass
#             else :
#                 raise ValueError("Unexpected node type: '%s'" % nodetype)
#
#         for c in source.constraints() :
#             destination.addConstraint(c)
#
#         destination.setNamesWithLabel(source.getNamesWithLabel())
#         destination.setWeights(source.getWeights())
#
#         destination.ready()
#         return destination


class CNF(LogicDAG):
    """A logic formula in Conjunctive Normal Form."""

    def __init__(self, **kwdargs):
        LogicDAG.__init__(self)
        self._clauses = []        # All clauses in the CNF (incl. comment)
        self._clausecount = 0     # Number of actual clauses (not incl. comment)

        self._weights = {}        # Weights of atoms in the CNF
        self._constraints = []  # Constraints

        # names, constraints, ...

    def add_atom(self, atom):
        self._atomcount += 1

    def add_comment(self, comment):
        self._clauses.append(['c', comment])

    def add_clause(self, head, body):
        self._clauses.append([head] + list(body))
        self._clausecount += 1

    def add_constraint(self, constraint):
        """Add a constraint.

        :param constraint:
        :type constraint: Constraint
        :return:
        """
        self._constraints.append(constraint)
        for c in constraint.encodeCNF():
            self.add_clause(None, c)

    def _clause2str(self, clause, weighted=False):
        if weighted:
            raise NotImplementedError()
        else:
            if clause[1] is None:
                return ' '.join(map(str, clause[2:])) + ' 0'
            else:
                return ' '.join(map(str, clause[1:])) + ' 0'

    def to_dimacs(self, partial=False, weighted=False, semiring=None):
        if weighted:
            t = 'wcnf'
        else:
            t = 'cnf'

        header, content = self.contents(partial=partial, weighted=weighted, semiring=semiring)

        result = 'p %s %s\n' % (t, ' '.join(map(str, header)))
        result += '\n'.join(map(lambda cl: ' '.join(map(str, cl)) + ' 0', content))
        return result

    def contents(self, partial=False, weighted=False, semiring=None):
        # Helper function to determine the certainly true / possibly true names (for partial)

        ct = lambda i: 2*i
        pt = lambda i: ct(i) - 1
        cpt = lambda i: -pt(-i) if i < 0 else ct(i)

        w_mult = 1
        w_max = []
        weights = None
        if weighted == int:
            w_mult = 10000
            wt = lambda w: int(w * w_mult)
        elif weighted == float:
            w_mult = 1
            wt = lambda w: w
        elif weighted:
            w_mult = 10000
            wt = lambda w: int(w * w_mult)

        if weighted:
            if semiring is None:
                semiring = SemiringLogProbability()
            weights = self.extractWeights(semiring)

            w_sum = 0.0
            for w_pos, w_neg in weights.values():
                w_sum += w_pos + w_neg
            w_max = [int(-w_sum*w_mult) + 1]

        atomcount = self.atomcount
        if partial:
            atomcount *= 2
        clausecount = self.clausecount
        if partial:
            clausecount += atomcount

        clauses = []

        if partial:
            # For each atom: add constraint
            for a in range(1, self.atomcount+1):
                clauses.append(w_max + [pt(a), -ct(a)])

                if weighted:
                    w_pos, w_neg = weights.get(a, (semiring.one(), semiring.one()))
                    if not semiring.is_one(w_pos):
                        clauses.append([-wt(w_pos), -ct(a)])
                    if not semiring.is_one(w_neg):
                        clauses.append([-wt(w_neg), pt(a)])

            # For each clause:
            for c in self.clauses:
                head, body = c[0], c[1:]
                if head is not None:
                    # Clause does not represent a constraint.
                    head_neg = (head < 0)
                    head = abs(head)
                    head1, head2 = ct(head), pt(head)
                    if head_neg:
                        head1, head2 = -head1, -head2
                    clauses.append(w_max + [head1, head2] + list(map(cpt, body)))
                else:
                    clauses.append(w_max + list(map(cpt, body)))
        else:
            if weighted:
                for a in range(1, self.atomcount+1):
                    w_pos, w_neg = weights.get(a, (semiring.one(), semiring.one()))
                    if not semiring.is_one(w_pos):
                        clauses.append([-wt(w_pos), -a])
                    if not semiring.is_one(w_neg):
                        clauses.append([-wt(w_neg), a])
            for c in self.clauses:
                head, body = c[0], c[1:]
                if head is not None:
                    clauses.append(w_max + [head] + list(body))
                else:
                    clauses.append(w_max + list(body))

        return [atomcount, len(clauses)] + w_max, clauses

    def constraints(self):
        return self._constraints

    def set_weights(self, weights):
        self._weights = weights

    def get_weights(self):
        return self._weights

    def is_trivial(self):
        return self.clausecount == 0

    @property
    def clauses(self):
        return self._clauses

    @property
    def clausecount(self):
        return self._clausecount


class PartialCNF(CNF):

    def __init__(self):
        CNF.__init__(self)
        self.is_partial = True







@transform(LogicDAG, CNF)
def clarks_completion(source, destination, **kwdargs):

    with Timer('Clark\'s completion'):
        # Each rule in the source formula will correspond to an atom.
        num_atoms = len(source)

        # Copy weight information.
        destination.set_weights(source.get_weights())

        # Add atoms.
        for i in range(0, num_atoms):
            destination.add_atom(i+1)

        # Complete other nodes
        # Note: assumes negation is encoded as negative number.
        for index, node, nodetype in source:
            if nodetype == 'conj':
                destination.add_clause(index, list(map(lambda x: -x, node.children)))
                for c in node.children:
                    destination.add_clause(-index, [c])
            elif nodetype == 'disj':
                destination.add_clause(-index, node.children)
                for c in node.children:
                    destination.add_clause(index, [-c])
            elif nodetype == 'atom':
                pass
            else:
                raise ValueError("Unexpected node type: '%s'" % nodetype)

        # Copy constraints.
        for c in source.constraints():
            destination.add_constraint(c)

        # Copy node names.
        for n, i, l in source.get_names_with_label():
            destination.addName(n, i, l, external=True)

        return destination
