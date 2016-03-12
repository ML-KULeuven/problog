#! /usr/bin/env python3
# coding=utf-8

"""
Example on how to use multiplicative factorization in ProbLog.

Observations:
- Only for large (>30) disjunctions, it results in a speedup.
- For really large (>80) disjunctions, the speedup is an order of magnitude
"""

from __future__ import print_function

import sys, os
import logging
import math
try:
    from itertools import zip_longest
except:
    from itertools import izip_longest as zip_longest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from problog.program import PrologFile
from problog.logic import Term
from problog.engine import DefaultEngine
from problog.formula import LogicFormula, LogicDAG
from problog.sdd_formula import SDD
from problog.nnf_formula import NNF
from problog.cnf_formula import CNF
from problog.forward import ForwardInference, ForwardBDD
from problog.util import Timer, start_timer, stop_timer, init_logger, format_dictionary
from problog.constraint import ClauseConstraint, ConstraintAD
from problog.evaluator import SemiringProbability, FormulaEvaluator, OperationNotSupported

def multiplicative_factorization(source, disjunct_threshold=8, disjunct_max=None):
    """
    Copy the source LogicFormula and replace large disjunctions with Skolemization variables
    to obtain a formula that is similar to the result of multiplicative factorization.

    Based on:
    - M. Takikawa and B. D'Ambrosio. Multiplicative factorization of noisy-max. In K. B. Laskey and H. Prade, editors, Proceedings of the 15th Conference on Uncertainty in Artificial Intelligence (UAI), pages 622-630, 1999.
    - F. J. Diez and S. F. Galan. Efficient computation for the noisy max. International Journal of Intelligent Systems, 18(2):165-177, 2003.
    - G. Van den Broeck, W. Meert, and A. Darwiche. Skolemization for weighted first-order model counting. In Proceedings of the 14th International Conference on Principles of Knowledge Representation and Reasoning (KR), 2014.
    - W. Meert, G. Van den Broeck, and A. Darwiche. Lifted inference for probabilistic logic programs. In Proceedings of the Workshop on Probabilistic Logic Programming (PLP), pages 1-8, July 2014.

    Transformation with disjunct_threshold<3 and disjunct_max=1 (following [Van den Broeck 2015]):

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

    :param source: Input formula
    :param disjunct_threshold: Split disjuncts larger than this threshold
    :param disjunct_max: Maximal size of new disjuncts (default is disjunct_threshold)
    :return: new formula, additional query atoms
    """
    assert(type(source) == LogicFormula) # Can be relaxed to isinstance
    logger = logging.getLogger('problog')
    target = LogicFormula()
    tseitin_vars = []
    skolem_vars = []
    extra_clauses = []
    extra_queries = []
    identifier_prefix = 'MF_'
    if disjunct_max is None:
        disjunct_max = disjunct_threshold

    logger.info('Disjunction threshold: {}'.format(disjunct_threshold))
    logger.info('Disjunction max block size: {}'.format(disjunct_max))

    # Copy formulas and replace large disjunctions with a Tseitin variable z_i
    for key,node,t in source:
        logger.debug('Rule: {:<3}: {} -- {}'.format(key,node,t))
        nodetype = type(node).__name__
        if nodetype == 'disj':
            if len(node.children) > disjunct_threshold:
                # Replace disjunction node with atom node.
                tseitin_vars.append(Term(identifier_prefix+'z_{}'.format(len(tseitin_vars))))
                logger.debug('-> {}: replace disj with tseitin {}'.format(key, tseitin_vars[-1]))
                skolem_vars.append(Term(identifier_prefix+'s_{}'.format(len(skolem_vars))))
                tseitin = target.add_atom(identifier=str(tseitin_vars[-1]), probability=(1.0,1.0), name=tseitin_vars[-1])
                # TODO: the name is overridden
                logger.info('Tseitin variable {} for {} children'.format(tseitin_vars[-1], len(node.children)))
                extra_clauses.append((tseitin, tseitin_vars[-1], skolem_vars[-1], node.children))
            else:
                target.add_or(components=node.children, key=key, name=node.name)
        elif nodetype == 'conj':
            target.add_and(components=node.children, key=key, name=node.name)
        elif nodetype == 'atom':
            # TODO: key and identifier are not the same! + what is the structure?
            target.add_atom(identifier=node.identifier, probability=node.probability, name=node.name,
                            group=node.group, source=node.source)
        else:
            raise Exception('Unknown type when performing multiplicative factorization:\n'
                            '{:<2}: {} ({})'.format(key, node, t))


    # Add Skolem and Tseitin clauses
    cur_body = 0
    all_top = [len(target)]
    for tseitin, tseitin_var, skolem_var, children in extra_clauses:
        logger.debug('-> Add tseitin {}'.format(tseitin_var))
        skolem = target.add_atom(identifier=str(skolem_var), probability=(1.0,-1.0), name=skolem_var)
        logger.debug('-> Add skolem {}'.format(skolem_var))
        target.add_name(skolem_var, skolem, target.LABEL_QUERY)
        extra_queries.append(skolem_var)
        target.add_constraint(ClauseConstraint([tseitin, skolem]))

        disjunct_size = math.ceil(len(children)/math.ceil(len(children)/disjunct_max))
        for c_group in zip_longest(*([iter(children)]*disjunct_size), fillvalue=None):
            c_group = [c for c in c_group if c is not None]
            logger.debug('-> Skolem group {}'.format(c_group))
            term = Term(identifier_prefix+'b_{}'.format(cur_body))
            cur_body += 1
            if len(c_group) == 1:
                d = c_group[0]
            else:
                d = target.add_or(components=c_group, key=None, name=term)
            target.add_name(term, d, target.LABEL_QUERY)
            extra_queries.append(term)
            target.add_constraint(ClauseConstraint([tseitin, -d]))
            target.add_constraint(ClauseConstraint([skolem,  -d]))


    # Copy labels
    for name, key, label in source.get_names_with_label():
        logger.debug('Name: {:<3} {} -> {}'.format(key, name, label))
        target.add_name(name, key, label)

    # Copy constraints
    for c in source.constraints():
        if not isinstance(c, ConstraintAD): # TODO correct?
            target.add_constraint(c)
            logger.debug('Constraint: {}'.format(c))
        else:
            logger.debug('Constraint(ignored): {}'.format(c))

    logger.debug('----- After multiplicative factorization -----')
    for key,node,t in target:
        logger.debug('Rule: {:<3}: {} -- {}'.format(key,node,t))
    for c in target.constraints():
        logger.debug('constraint: {}'.format(c))
    for q, n in target.queries():
        logger.debug('query: {} {}'.format(q,n))
    for q, n, v in target.evidence_all():
        logger.debug('evidence: {} {} {}'.format(q,n,v))
    for name, key, label in target.get_names_with_label():
        logger.debug('Name: {:<3} {} -> {}'.format(key, name, label))
    logger.debug('-----\n')

    return target, extra_queries

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
        print('normalize {} {}'.format(a,z))
        if isinstance(a, tuple) or isinstance(z, tuple):
            raise OperationNotSupported()
        if -0.0001 < a < 0.0001:
            return a
        return a / z

    def pos_value(self, a, key=None):
        if isinstance(a, tuple):
            return self.value(a[0])
        return self.value(a)

    def neg_value(self, a, key=None):
        if isinstance(a, tuple):
            return self.value(a[1])
        else:
            return self.negate(self.value(a))

    def value(self, a):
        v = float(a)
        return v

def probability(filename, with_fact=True, knowledge='nnf', disjunct_threshold=8, disjunct_max=None):
    logger = logging.getLogger('problog')
    pl = PrologFile(filename)
    engine = DefaultEngine(label_all=True, keep_all=True, keep_duplicates=True)
    db = engine.prepare(pl)
    gp = engine.ground_all(db)
    # if logger.isEnabledFor(logging.DEBUG):
        # logger.debug(gp.to_prolog())
    semiring = NegativeProbability()

    if with_fact:
        with Timer('ProbLog with multiplicative factorization'):
            if logger.isEnabledFor(logging.DEBUG):
                with open('test_f.dot', 'w') as dotfile:
                    print_result((True, gp.to_dot()), output=dotfile)
                cnf = CNF.createFrom(gp)
                # logger.debug(cnf.to_dimacs())
            gp2, extra_queries = multiplicative_factorization(gp, disjunct_threshold, disjunct_max)
            if logger.isEnabledFor(logging.DEBUG):
                with open('test_f_mf.dot', 'w') as dotfile:
                    print_result((True, gp2.to_dot()), output=dotfile)
                gp_acyclic = LogicDAG.createFrom(gp2)
                with open('test_f_mf_acyclic.dot', 'w') as dotfile:
                    print_result((True, gp_acyclic.to_dot()), output=dotfile)
            with Timer('Compilation with {}'.format(knowledge)):
                if knowledge == 'sdd':
                    logger.warning("SDDs not yet fully supported")
                    nnf = SDD.create_from(gp2)
                    ev = nnf.to_formula()
                    fe = FormulaEvaluator(ev, semiring)
                    weights = ev.extract_weights(semiring=semiring)
                    fe.set_weights(weights)
                    # TODO: SDD lib doesn't support negative weights? Runtime error
                    # TODO: How can I use the Python evaluator with SDDs?
                elif knowledge == 'fbdd':
                    logger.warning("fbdd not yet fully supported")
                    # TODO: Stupid approach because it introduces new root levels
                    nnf = ForwardBDD.createFrom(gp2)
                else:
                    nnf = NNF.createFrom(gp2)
            if logger.isEnabledFor(logging.DEBUG):
                with open ('test_f_mf_nnf.dot', 'w') as dotfile:
                    print(nnf.to_dot(), file=dotfile)
                cnf = CNF.createFrom(gp2)
                # logger.debug(cnf.to_dimacs())
            logger.debug('Deleting queries: {}'.format(extra_queries))
            for query in extra_queries:
                nnf.del_name(query, nnf.LABEL_QUERY)
            logger.info('NNF stats:\n'+'\n'.join(['{:<6}: {:>10,}'.format(k,v) for k,v in sorted(nnf.stats().items())]))
            with Timer('Evalation'):
                result = nnf.evaluate(semiring=semiring)
    else:
        with Timer('ProbLog without multiplicative factorization'):
            if logger.isEnabledFor(logging.DEBUG):
                with open('test_nf.dot', 'w') as dotfile:
                    print_result((True, gp.to_dot()), output=dotfile)
            with Timer('Compilation with {}'.format(knowledge)):
                if knowledge == 'sdd':
                    nnf = SDD.createFrom(gp)
                elif knowledge == 'fbdd':
                    nnf = ForwardBDD.createFrom(gp)
                else:
                    nnf = NNF.createFrom(gp)
            # with open ('test_nf_nnf.dot', 'w') as dotfile:
            #     print(nnf.to_dot(), file=dotfile)
            logger.info('NNF stats:\n'+'\n'.join(['{:<6}: {:>10,}'.format(k,v) for k,v in sorted(nnf.stats().items())]))
            with Timer('Evaluation'):
                result = nnf.evaluate(semiring=semiring)
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
    parser.add_argument('--profile', action='store_true', help='Profile Python script')
    parser.add_argument('--threshold', '-t', default=30, type=int,
                        help='Threshold to break disjunctions')
    parser.add_argument('--disjunctblocksize', '-b', default=1, type=int,
                        help='Block size of splits of disjunct')
    parser.add_argument('--knowledge', '-k', default='nnf',
                        help='Knowledge compilation (sdd, nnf, fbdd)')
    args = parser.parse_args()

    init_logger(args.verbose)

    if args.profile:
        import cProfile, pstats
        cProfile.run('probability( args.filename, not args.nomf, args.knowledge, args.threshold, args.disjunctblocksize)', 'prstats')
        p = pstats.Stats('prstats')
        p.strip_dirs()
        # p.sort_stats('cumulative')
        p.sort_stats('time')
        p.print_stats()
    else:
        result = probability( args.filename, not args.nomf, args.knowledge, args.threshold, args.disjunctblocksize)
        print_result((True,result), sys.stdout)
