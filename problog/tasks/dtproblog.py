#! /usr/bin/env python

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
import logging
import traceback

from ..program import PrologFile
from ..engine import DefaultEngine
from ..logic import Term
from ..errors import process_error, ProbLogError
from .. import get_evaluatables, get_evaluatable
from ..util import init_logger, Timer, format_dictionary


def main(argv, result_handler=None):
    args = argparser().parse_args(argv)
    inputfile = args.inputfile

    init_logger(args.verbose, name='dtproblog')

    if result_handler is None:
        if args.web:
            result_handler = print_result_json
        else:
            result_handler = print_result

    if args.output is not None:
        outf = open(args.output, 'w')
    else:
        outf = sys.stdout

    try:
        model = PrologFile(inputfile)  # factory=factory
        result = dtproblog(model, **vars(args))

        choices, score, stats = result
        logging.getLogger('dtproblog').info('Number of strategies evaluated: %s' % stats.get('eval'))

        renamed_choices = {}
        for k, v in choices.items():
            if k.functor == 'choice':
                k = k.args[2]
            renamed_choices[k] = v

        result_handler((True, (renamed_choices, score, stats)), outf)
    except Exception as err:
        err.trace = traceback.format_exc()
        result_handler((False, err), outf)

    if args.output is not None:
        outf.close()


def dtproblog(model, search=None, koption=None, locations=False, web=False, **kwargs):
    """Evaluate a DT ProbLog model

    :param model: ProbLog model
    :type model: problog.logic.LogicProgram
    :param search: specifies search ('exhaustive' or 'local')
    :param koption: specifies knowledge compilation tool (omit for system default)
    :param locations: add Term locations to results
    :param web: prepare for web mode
    :param kwargs: additional arguments (passed to search procedure)
    :return: best decisions, score of best decision, statistics
    """

    with Timer('Total', logger='dtproblog'):
        with Timer('Parse input', logger='dtproblog'):
            eng = DefaultEngine(label_all=True)
            db = eng.prepare(model)

        with Timer('Ground', logger='dtproblog'):
            # decisions = dict((d[0], None) for d in eng.query(db, Term('decision', None)))
            utilities = dict(eng.query(db, Term('utility', None, None)))

            # logging.getLogger('dtproblog').debug('Decisions: %s' % decisions)
            logging.getLogger('dtproblog').debug('Utilities: %s' % utilities)

            # for d in decisions:
            #     db += d.with_probability(Constant(0.5))

            gp = eng.ground_all(db, target=None, queries=utilities.keys())
            decisions = []
            decision_nodes = set()
            for i, n, t in gp:
                if t == 'atom' and n.probability == Term('?'):
                    decisions.append((i, n.name))
                    decision_nodes.add(i)

            constraints = []
            for c in gp.constraints():
                if set(c.get_nodes()) & decision_nodes:
                    constraints.append(c)

        with Timer('Compile', logger='dtproblog'):
            knowledge = get_evaluatable(koption).create_from(gp)

        with Timer('Optimize', logger='dtproblog'):
            if search == 'local':
                result = search_local(knowledge, decisions, utilities, constraints, **kwargs)
            else:
                result = search_exhaustive(knowledge, decisions, utilities, constraints, **kwargs)

        if web or locations:
            for k, v in result[0].items():
                if k.functor == 'choice':
                    k.args[2].loc = db.lineno(k.args[2].location)
                else:
                    k.loc = db.lineno(k.location)

    return result


def evaluate(formula, decisions, utilities):
    result = formula.evaluate(weights=decisions)

    score = 0.0
    for r in result:
        score += result[r] * float(utilities[r])
    return score


def search_exhaustive(formula, decisions, utilities, constraints, verbose=0, **kwargs):
    stats = {'eval': 0}
    best_score = None
    best_choice = None

    decision_ids, decision_names = zip(*decisions)

    for i in range(0, 1 << len(decisions)):
        choices = num2bits(i, len(decisions))

        evidence = dict(zip(decision_names, map(int, choices)))

        constraints_ok = True
        for c in constraints:
            if not c.check(dict(zip(decision_ids, map(int, choices)))):
                constraints_ok = False
                break
        if not constraints_ok:
            continue

        score = evaluate(formula, evidence, utilities)
        stats['eval'] += 1
        if best_score is None or score > best_score:
            best_score = score
            best_choice = dict(evidence)
            logging.getLogger('dtproblog').debug('Improvement: %s -> %s' % (best_choice, best_score))
    return best_choice, best_score, stats


def search_local(formula, decisions, utilities, constraints, verbose=0, **kwargs):
    """Performs local search.

    :param formula:
    :param decisions:
    :param utilities:
    :param verbose:
    :param kwargs:
    :return:
    """
    stats = {'eval': 1}

    for c in constraints:
        if not c.is_true():
            raise ProbLogError('Local search does not support constraints')

    choices = {}
    # Create the initial strategy:
    #  for each decision, take option that has highest local utility
    #   (takes false if no utility is given for the decision variable)
    for ident, key in decisions:
        if key in utilities and float(utilities[key]) > 0:
            choices[key] = 1
        else:
            choices[key] = 0

    # Compute the score of the initial strategy.
    best_score = evaluate(formula, choices, utilities)

    # Perform local search by flipping one decision at a time
    last_update = None  # Last decision that was flipped and improved the score
    stop = False
    while not stop:
        # This loop stops when either of these conditions is met:
        #   - at the end of the (first) iteration no decision was flipped successfully
        #   - while iterating we again reach the last decision that was flipped
        #       (this means we tried to flip all decisions, but none were successfull)
        for ident, key in decisions:
            if last_update == key:
                # We went through all decisions without flipping since the last flip.
                stop = True
                break
            # Flip a decision
            choices[key] = 1 - choices[key]
            # Compute the score of the new strategy
            flip_score = evaluate(formula, choices, utilities)
            stats['eval'] += 1
            if flip_score <= best_score:
                # The score is not better: undo the flip
                choices[key] = 1 - choices[key]
            else:
                # The score is better: update best score and pointer to last_update
                last_update = key
                best_score = flip_score
                logging.getLogger('dtproblog').debug('Improvement: %s -> %s' % (choices, best_score))
        if last_update is None:
            # We went through all decisions without flipping.
            stop = True

    return choices, best_score, stats


def num2bits(n, nbits):
    bits = [False] * nbits
    for i in range(1, nbits + 1):
        bits[nbits - i] = bool(n % 2)
        n >>= 1
    return bits


def print_result(result, output=sys.stdout):
    success, result = result
    if success:
        choices, score, stats = result
        print(format_dictionary(choices, 0), file=output)
        print('SCORE: %s' % score, file=output)
        return 0
    else:
        print(process_error(result), file=output)
        return 1


def print_result_json(d, output):
    """Pretty print result.

    :param d: result from run_problog
    :param output: output file
    :return:
    """
    import json
    result = {}
    success, d = d
    if success:
        choices, score, stats = d
        result['SUCCESS'] = True

        resout = []
        for n, p in choices.items():
            if hasattr(n, 'loc') and n.loc is not None:
                resout.append([str(n), int(p), n.loc[1], n.loc[2]])
            else:
                resout.append([str(n), int(p), None, None])
        result['choices'] = resout
        result['score'] = score
        result['stats'] = stats
    else:
        result['SUCCESS'] = False
        result['err'] = vars(d)
        result['err']['message'] = process_error(result)
    print (json.dumps(result), file=output)
    return 0


# class DTProbLogFactory(ExtendedPrologFactory):
#
#     def build_probabilistic(self, operand1, operand2, location=None, **extra):
#         if str(operand1.functor) in 'd?':
#             return Term('decision', operand2, location=location)
#         else:
#             return ExtendedPrologFactory.build_probabilistic(self, operand1, operand2, location, **extra)


def argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    parser.add_argument('--knowledge', '-k', dest='koption',
                        choices=get_evaluatables(),
                        default=None, help="Knowledge compilation tool.")
    parser.add_argument('-s', '--search', choices=('local', 'exhaustive'), default='exhaustive')
    parser.add_argument('-v', '--verbose', action='count', help='Set verbosity level')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Write output to given file (default: write to stdout)')
    parser.add_argument('--web', action='store_true', help=argparse.SUPPRESS)
    return parser


if __name__ == '__main__':
    main(sys.argv[1:])
