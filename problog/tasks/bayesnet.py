"""
ProbLog command-line interface.

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
import itertools
import logging

from problog.formula import LogicDAG
from problog.parser import DefaultPrologParser
from problog.program import ExtendedPrologFactory, PrologFile
from problog.logic import Term, Or, Clause, And, Not
from problog.pgm.cpd import Variable, Factor, OrCPT, PGM

from ..util import init_logger
from ..errors import process_error


def print_result_standard(result, output=sys.stdout):
    """Pretty print result.

    :param result: result from run_problog
    :param output: output file
    :return:
    """
    success, result = result
    if success:
        print(result, file=output)
        return 0
    else:
        print(process_error(result), file=output)
        return 1


def term_to_atoms(terms):
    """Visitor to list atoms in term."""
    if not isinstance(terms, list):
        terms = [terms]
    new_terms = []
    for term in terms:
        if isinstance(term, And):
            new_terms += term_to_atoms(term.to_list())
        elif isinstance(term, Or):
            new_terms += term_to_atoms(term.to_list())
        elif isinstance(term, Not):
            new_terms.append(term.child)
        else:
            new_terms.append(term)
    return new_terms


def term_to_bool(term, truth_values):
    """Visitor that substitutes atoms with truth values.

    :param: truth_values: Dictionary atom -> True/False
    :return: term
    """
    logger = logging.getLogger('problog')
    if isinstance(term, And):
        op1 = term_to_bool(term.op1, truth_values)
        if op1 is False:
            return False
        elif op1 is True:
            op2 = term_to_bool(term.op2, truth_values)
            if op2 is False:
                return False
            elif op2 is True:
                return True
            else:
                logger.error('and.op2 is not a boolean'.format(op2))
                return None
        else:
            logger.error('and.op1 is not a boolean'.format(op1))
            return None
    elif isinstance(term, Or):
        op1 = term_to_bool(term.op1, truth_values)
        if op1 is True:
            return True
        elif op1 is False:
            op2 = term_to_bool(term.op2, truth_values)
            if op2 is True:
                return True
            elif op2 is False:
                return False
            else:
                logger.error('or.op2 is not a boolean'.format(op2))
                return None
        else:
            logger.error('or.op1 is not a boolean'.format(op1))
            return None
    elif isinstance(term, Not):
        child = term_to_bool(term.child, truth_values)
        if child is True:
            return False
        elif child is False:
            return True
        else:
            logger.error('not.child is not a boolean: {}'.format(child))
            return None
    else:
        if term in truth_values:
            return truth_values[term]
        else:
            logger.error('Unknown term: {} ({})'.format(term, type(term)))
            return None


def clause_to_cpt(clause, number, pgm):
    logger = logging.getLogger('problog')
    # print('Clause: {} -- {}'.format(clause, type(clause)))
    if isinstance(clause, Clause):
        if isinstance(clause.head, Or):
            heads = clause.head.to_list()
        elif isinstance(clause.head, Term):
            heads = [clause.head]
        else:
            heads = []
            logger.error('Unknown head type: {} ({})'.format(clause.head,
                                                             type(clause.head)))
        # CPT for the choice node
        rv_cn = 'c{}'.format(number)
        parents = term_to_atoms(clause.body)
        parents_str = [str(t) for t in parents]
        probs_heads = []
        for head in heads:
            if head.probability is not None:
                probs_heads.append(head.probability.compute_value())
            else:
                probs_heads.append(1.0)
        probs = [1.0-sum(probs_heads)]+probs_heads
        table_cn = dict()
        for keys in itertools.product([False, True], repeat=len(parents)):
            truth_values = dict(zip(parents, keys))
            truth_value = term_to_bool(clause.body, truth_values)
            if truth_value is None:
                logger.error('Expected a truth value. Instead god:\n'
                             ' {} -> {}'.format(truth_values, truth_value))
                table_cn[keys] = [1.0] + [0.0]*len(heads)
            elif truth_value:
                table_cn[keys] = probs
            else:
                table_cn[keys] = [1.0] + [0.0]*len(heads)
        pgm.add_var(Variable(rv_cn, list(range(len(heads)+1))))
        cpd_cn = Factor(pgm, rv_cn, parents_str, table_cn)
        cpd_cn.latent = True
        pgm.add_factor(cpd_cn)
        # CPT for the head random variable
        for idx, head in enumerate(heads):
            rv = str(head.with_probability())
            pgm.add_var(Variable(rv, [0, 1]))
            pgm.add_factor(OrCPT(pgm, rv, [(rv_cn, idx+1)]))
        return

    elif isinstance(clause, Or):
        heads = clause.to_list()
        # CPT for the choice node
        rv_cn = 'c{}'.format(number)
        parents = []
        probs_heads = [head.probability.compute_value() for head in heads]
        table_cn = [1.0-sum(probs_heads)]+probs_heads
        pgm.add_var(Variable(rv_cn, list(range(len(heads)+1))))
        cpd_cn = Factor(pgm, rv_cn, parents, table_cn)
        cpd_cn.latent = True
        pgm.add_factor(cpd_cn)
        # CPT for the head random variable
        for idx, head in enumerate(heads):
            rv = str(head.with_probability())
            pgm.add_var(Variable(rv, [0, 1]))
            pgm.add_factor(OrCPT(pgm, rv, [(rv_cn, idx+1)]))
        return

    elif isinstance(clause, Term):
        # CPT for the choice node
        rv_cn = 'c{}'.format(number)
        parents = []
        prob = clause.probability.compute_value()
        table_cn = [1.0-prob, prob]
        pgm.add_var(Variable(rv_cn, [0, 1]))
        cpd_cn = Factor(pgm, rv_cn, parents, table_cn)
        cpd_cn.latent = True
        pgm.add_factor(cpd_cn)
        # CPT  for the head random variable
        rv = str(clause.with_probability())
        pgm.add_var(Variable(rv, [0, 1]))
        pgm.add_factor(OrCPT(pgm, rv, [(rv_cn, 1)]))
        return

    return


def formula_to_bn(formula):
    # factors = []
    # for i, n, t in formula:
    #     factors.append('{}: {}'.format(i, n))
    #     factors.append('{}: {}'.format(i, t))

    logger = logging.getLogger('problog')
    clauses = []
    bn = PGM()
    for idx, clause in enumerate(formula.enum_clauses()):
        logger.debug('clause: {}'.format(clause))
        clauses.append(clause)
        clause_to_cpt(clause, idx, bn)

    # lines = ['{}'.format(c) for c in clauses]
    # for qn, qi in formula.queries():
    #     if is_ground(qn):
    #         if qi == formula.TRUE:
    #             lines.append('%s.' % qn)
    #         elif qi == formula.FALSE:
    #             lines.append('%s :- fail.' % qn)
    #         lines.append('query(%s).' % qn)
    #
    # for qn, qi in formula.evidence():
    #     if qi < 0:
    #         lines.append('evidence(%s).' % -qn)
    #     else:
    #         lines.append('evidence(%s).' % qn)

    # bn.marginalizeLatentVariables()

    return bn


def argparser():
    """Argument parser for transformation to Bayes net.
    :return: argument parser
    :rtype: argparse.ArgumentParser
    """
    import argparse
    description = """Translate a ProbLog program to a Bayesian network.
    """
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('filename', metavar='MODEL', type=str, help='input ProbLog model')
    parser.add_argument('--verbose', '-v', action='count', help='Verbose output')
    parser.add_argument('-o', '--output', type=str, help='output file', default=None)
    parser.add_argument('--format', choices=('hugin', 'xdsl', 'uai08', 'dot', 'internal'), default=None,
                        help='Output format')
    parser.add_argument('--keep-all', action='store_true', help='also output deterministic nodes')
    # parser.add_argument('--keep-duplicates', action='store_true', help='don\'t eliminate duplicate literals')
    parser.add_argument('--hide-builtins', action='store_true', help='hide deterministic part based on builtins')
    return parser


def main(argv, result_handler=None):
    parser = argparser()
    args = parser.parse_args(argv)
    init_logger(args.verbose)

    outfile = sys.stdout
    if args.output:
        outfile = open(args.output, 'w')
    target = LogicDAG  # Break cycles
    print_result = print_result_standard

    try:
        gp = target.createFrom(
            PrologFile(args.filename, parser=DefaultPrologParser(ExtendedPrologFactory())),
            label_all=True, avoid_name_clash=False, keep_order=True,  # Necessary for to prolog
            keep_all=args.keep_all, keep_duplicates=False,  # args.keep_duplicates,
            hide_builtins=args.hide_builtins)
        # rc = print_result((True, gp), output=outfile)
        # p = PrologFile(args.filename)
        # engine = DefaultEngine(label_all=True, avoid_name_clash=True, keep_order=True,
        #                        keep_duplicates=False, keep_all=True)
        # db = engine.prepare(p)
        # gp = engine.ground_all(db)

        bn = formula_to_bn(gp)
        if args.format == 'hugin':
            bn_str = bn.to_hugin_net()
        elif args.format == 'xdsl':
            bn_str = bn.to_xdsl()
        elif args.format == 'uai08':
            bn_str = bn.to_uai08()
        elif args.format == 'dot':
            bn_str = bn.to_graphviz()
        else:
            bn_str = str(bn)
        rc = print_result((True, bn_str), output=outfile)

    except Exception as err:
        import traceback
        err.trace = traceback.format_exc()
        rc = print_result((False, err))

    if args.output:
        outfile.close()

    if rc:
        sys.exit(rc)


if __name__ == '__main__':
    main(sys.argv[1:])
