#! /usr/bin/env python
# coding=utf-8

"""
Example on how to use multiplicative factorization in ProbLog.

Based on:
- M. Takikawa and B. D'Ambrosio. Multiplicative factorization of noisy-max. In K. B. Laskey and H. Prade, editors, Proceedings of the 15th Conference on Uncertainty in Artificial Intelligence (UAI), pages 622-630, 1999.
- F. J. Diez and S. F. Galan. Efficient computation for the noisy max. International Journal of Intelligent Systems, 18(2):165-177, 2003.
- G. Van den Broeck, W. Meert, and A. Darwiche. Skolemization for weighted first-order model counting. In Proceedings of the 14th International Conference on Principles of Knowledge Representation and Reasoning (KR), 2014.
- W. Meert, G. Van den Broeck, and A. Darwiche. Lifted inference for probabilistic logic programs. In Proceedings of the Workshop on Probabilistic Logic Programming (PLP), pages 1-8, July 2014.

Transformation (following [Van den Broeck 2015]):

… (c1 ∨ c2 ∨ c3) …
… ⋁ci …

(… z …)
∧ (z ∨ ¬⋁ci)
∧ (z ∨ s)
∧ (s ∨ ¬⋁ci)

(… z …)
∧ (z ∨ ¬c1)
∧ (z ∨ ¬c2)
∧ (z ∨ ¬c3)
∧ (z ∨ s)
∧ (s ∨ ¬c1)
∧ (s ∨ ¬c2)
∧ (s ∨ ¬c3)

with w(z) = w(¬z) = w(s) = 1 and w(¬s) = -1
"""

from __future__ import print_function

import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from problog.program import PrologFile
from problog.logic import Term
from problog.engine import DefaultEngine
from problog.formula import LogicFormula
from problog.sdd_formula import SDD
from problog.nnf_formula import NNF
from problog.util import Timer, start_timer, stop_timer, init_logger, format_dictionary
from problog.constraint import ClauseConstraint, ConstraintAD
from problog.evaluator import SemiringProbability, OperationNotSupported


def multfact(source):
    target = LogicFormula()
    tseitin_vars = []
    skolem_vars = []
    extra_clauses = []

    for key,node,t in source:
        print('Rule: {:<3}: {} -- {}'.format(key,node,t))
        nodetype = type(node).__name__
        # print(node.name)
        # print(type(node.name))
        # print(nodetype)
        if nodetype == 'disj':
            if len(node.children) > 1: #4:
                # Replace disjunction node with atom node.
                tseitin_vars.append(Term('z_{}'.format(len(tseitin_vars))))
                print('-> {}: replace disj with tseitin {}'.format(key, tseitin_vars[-1]))
                skolem_vars.append(Term('s_{}'.format(len(skolem_vars))))
                tseitin = target.add_atom(identifier=key, probability=(1.0,1.0), name=tseitin_vars[-1])
                target.add_name(tseitin_vars[-1], tseitin, target.LABEL_QUERY)
                # print('Added nodes: {} {} -- {} {}'.format(tseitin_vars[-1], tseitin, skolem_vars[-1], skolem))
                # print('Child nodes: {}'.format(node.children))
                extra_clauses += [(tseitin, skolem_vars[-1], node.children)]
            else:
                target.add_or(components=node.children, key=key, name=node.name)
        elif nodetype == 'conj':
            target.add_and(components=node.children, key=key, name=node.name)
        elif nodetype == 'atom':
            target.add_atom(identifier=key, probability=node.probability, name=node.name,
                            group=node.group, source=node.source)
        else:
            raise Exception('Unknown type when performing multiplicative factorization:\n'
                            '{:<2}: {} ({})'.format(key, node, t))

    for name, key, label in source.get_names_with_label():
        print('Name: {:<3} {} -> {}'.format(key, name, label))
        target.add_name(name, key, label)

    for c in source.constraints():
        if not isinstance(c, ConstraintAD): # TODO correct?
            target.add_constraint(c)
            print('Constraint: {}'.format(c))
        else:
            print('Constraint(ignored): {}'.format(c))

    # for q, n in source.queries():
    #     print('query: {} {}'.format(q,n))
    #     target.add_name(q, n, target.LABEL_QUERY)
    #
    # for q, n, v in source.evidence_all():
    #     print('evidence: {} {} {}'.format(q,n,v))
    #     target.add_name(q, n, v)

    # Add compensating clauses
    # ∧ (z ∨ ¬c1)
    # ∧ (z ∨ ¬c2)
    # ∧ (z ∨ ¬c3)
    # ∧ (z ∨ s)
    # ∧ (s ∨ ¬c1)
    # ∧ (s ∨ ¬c2)
    # ∧ (s ∨ ¬c3)
    cur_key = max([key for key,_,_ in target]) # TODO: more efficient way?
    # print('current key: {}'.format(cur_key))
    cur_body = 0
    for tseitin, skolem_var, children in extra_clauses:
        cur_key += 1
        skolem = target.add_atom(identifier=cur_key, probability=(1.0,-1.0), name=skolem_var)
        print('-> Add skolem {}'.format(skolem_var))
        target.add_name(skolem_var, skolem, target.LABEL_QUERY)
        target.add_constraint(ClauseConstraint([tseitin, skolem]))
        for c in children:
            target.add_name(Term('b_{}'.format(cur_body)), c, target.LABEL_QUERY)
            cur_body += 1
            target.add_constraint(ClauseConstraint([tseitin, -c]))
            target.add_constraint(ClauseConstraint([skolem, -c]))

    print('\n----- after multfact -----')
    for key,node,t in target:
        print('{:<2}: {} -- {}'.format(key,node,t))
    for c in target.constraints():
        print('constraint: {}'.format(c))
    for q, n in source.queries():
        print('query: {} {}'.format(q,n))
    for q, n, v in source.evidence_all():
        print('evidence: {} {} {}'.format(q,n,v))
    print('-----\n')

    return target

class NegativeProbability(SemiringProbability):
    """Represent negative probabilities (probably the same as just weights)."""

    @property
    def inconsistent_evidence_is_zero(self):
        return False

    def one(self):
        return 1.0

    def zero(self):
        return 0.0

    def is_one(self, value):
        if isinstance(value, tuple):
            raise OperationNotSupported()
        return 1.0 - 1e-12 < value < 1.0 + 1e-12

    def is_zero(self, value):
        if isinstance(value, tuple):
            raise OperationNotSupported()
        return -1e-12 < value < 1e-12

    def plus(self, a, b):
        if isinstance(a, tuple) or isinstance(b, tuple):
            raise OperationNotSupported()
        return a + b

    def times(self, a, b):
        if isinstance(a, tuple) or isinstance(b, tuple):
            raise OperationNotSupported()
        return a * b

    def negate(self, a):
        if isinstance(a, tuple):
            raise OperationNotSupported()
        return 1.0 - a

    def normalize(self, a, z):
        if isinstance(a, tuple) or isinstance(z, tuple):
            raise OperationNotSupported()
        return a / z

    def pos_value(self, a):
        # print('pos_value({})'.format(a))
        if isinstance(a, tuple):
            return self.value(a[0])
        return self.value(a)

    def neg_value(self, a):
        # print('neg_value({})'.format(a))
        if isinstance(a, tuple):
            return self.value(a[1])
        else:
            return self.negate(self.value(a))

    def value(self, a):
        # print('value({})'.format(a))
        v = float(a)
        return v

def probability(filename, with_fact=True):
    pl = PrologFile(filename)
    engine = DefaultEngine(label_all=True)#, keep_all=True, keep_duplicates=True)
    db = engine.prepare(pl)
    gp = engine.ground_all(db)
    print('type gp = '+str(type(gp)))

    semiring = NegativeProbability()

    print('-----')
    if with_fact:
        with Timer('With factorization'):
            gp2 = multfact(gp)
            with open('/Users/wannes/Desktop/test_f.dot', 'w') as dotfile:
                print_result((True, gp2.to_dot()), output=dotfile)
            print(gp2)
            # nnf = SDD.createFrom(gp2) # SDD lib doesn't support negative weights? Runtime error
            nnf = NNF.createFrom(gp2)
            print('--- start evaluating ---')
            print(nnf)
            with open ('/Users/wannes/Desktop/test_f_nnf.dot', 'w') as dotfile:
                print(nnf.to_dot(), file=dotfile)
            # TODO: propage should not try to simplify based on probs in this case
            result = nnf.evaluate(semiring=semiring)
            # TODO: problog evaluates graph to query, for skolemization we need P(x) = WMC(x=T)/WMC
    else:
        with Timer('No factorization'):
            with open('/Users/wannes/Desktop/test_nf.dot', 'w') as dotfile:
                print_result((True, gp.to_dot()), output=dotfile)
            # nnf = SDD.createFrom(gp)
            nnf = NNF.createFrom(gp)
            result = nnf.evaluate(semiring=semiring)

    print('-----')
    return result


def print_result( d, output, precision=8 ) :
    success, d = d
    if success :
        if not d : return 0 # no queries
        print(d, file=output)
        return 0
    else :
        print ('Error:', d, file=output)
        return 1

if __name__ == '__main__' :
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('filename')
    parser.add_argument('--verbose', '-v', action='count', help='Verbose output')
    parser.add_argument('--nomf', action='store_true', help='Disable multiplicative factorization')
    args = parser.parse_args()

    init_logger(args.verbose)

    result = probability( args.filename, not args.nomf)
    print_result((True,result), sys.stdout)
