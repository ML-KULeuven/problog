#! /usr/bin/env python
"""
Query-based sampler for ProbLog 2.1
===================================

  Concept:
      Each term has a value which is stored in the SampledFormula.
      When an probabilistic atom or choice atom is evaluated:
          - if probability is boolean discrete:
                          determine Yes/No, store in formula and return 0 (True) or None (False)
          - if probability is non-boolean discrete:
                          determine value, store value in formula and return key to value
      Adds builtin sample(X,S) that calls X and returns the sampled value in S.


  How to support evidence?
      => evidence on facts or derived from deterministic data: trivial, just fix value
      => evidence on atoms derived from other probabilistic facts:
          -> query evidence first
          -> if evidence is false, restart immediately
  Currently, evidence is supported through post-processing.


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

from problog.program import PrologFile
from problog.logic import Term, Constant, ArithmeticError
from problog.engine import DefaultEngine, UnknownClause
from problog.engine_builtin import check_mode
from problog.formula import LogicFormula
from problog.errors import process_error, GroundingError
from problog.util import start_timer, stop_timer, format_dictionary, init_logger
from problog.engine_unify import UnifyError, unify_value
import random
import math
import signal
import time
import traceback
import logging


try:

    import numpy.random

    def sample_poisson(l):
        l = list(l)
        return numpy.random.poisson(l)[0]

except ImportError:
    numpy = None

    def sample_poisson(l):
        l = list(l)[0]
        # http://www.johndcook.com/blog/2010/06/14/generating-poisson-random-values/
        if l >= 30:
            c = 0.767 - 3.36 / l
            beta = math.pi / math.sqrt(3.0 * l)
            alpha = beta * l
            k = math.log(c) - l - math.log(beta)

            while True:
                u = random.random()
                x = (alpha - math.log((1.0 - u) / u)) / beta
                n = math.floor(x + 0.5)
                if n < 0:
                    continue
                v = random.random()
                y = alpha - beta * x
                lhs = y + math.log(v / (1.0 + math.exp(y)) ** 2)
                rhs = k + n * math.log(l) - math.lgamma(n + 1)
                if lhs <= rhs:
                    return n
        else:
            l2 = math.exp(-l)
            k = 1
            p = 1
            while p > l2:
                k += 1
                p *= random.random()
            return k - 1





class FunctionStore(object):

    def __init__(self, target=None, engine=None, database=None):
        self.target = target
        self.engine = engine
        self.database = database

    # noinspection PyUnusedLocal
    def get(self, key, default=None):
        functor, arity = key

        def _function(*args):
            args = tuple(map(Constant, args))
            term = Term(functor, *args)
            try:
                results = self.engine.call(term, subcall=True, target=self.target,
                                           database=self.database)
                if len(results) == 1:
                    value_id = results[0][-1]
                    return float(self.target.get_value(value_id))
                elif len(results) == 0:
                    return None
                else:
                    raise ValueError('Unexpected multiple results.')
            except UnknownClause as err:
                sig = err.signature
                raise ArithmeticError("Unknown function %s" % sig)
        return _function


class SampledFormula(LogicFormula):

    def __init__(self, **kwargs):
        LogicFormula.__init__(self, **kwargs)
        self.facts = {}
        self.groups = {}
        self.probability = 1.0  # Try to compute
        self.values = []

        self.distributions = {
            'normal': random.normalvariate,
            'gaussian': random.normalvariate,
            'poisson': sample_poisson,
            'exponential': random.expovariate,
            'beta': random.betavariate,
            'gamma': random.gammavariate,
            'uniform': random.uniform,
            'triangular': random.triangular,
            'vonmises': random.vonmisesvariate,
            'weibull': random.weibullvariate,
        }

    def sample_value(self, term):
        """
        Sample a value from the distribution described by the term.
       :param term: term describing a distribution
       :type term: Term
       :return: value of the term, probability of the choice
       :rtype: Constant
        """
        if term.functor == 'fixed':
            return term.args[0], 1.0
        elif term.functor in self.distributions:
            args = map(float, term.args)
            value = self.distributions[term.functor](*args)
            return Constant(value), 0.0
        else:
            raise ValueError("Unknown distribution: '%s'" % term.functor)

    def _is_simple_probability(self, term):
        try:
            float(term)
            return True
        except ArithmeticError:
            return False

    def add_value(self, value):
        self.values.append(value)
        return len(self.values)

    def get_value(self, key):
        if key == 0:
            return None
        return self.values[key - 1]

    def add_atom(self, identifier, probability, group=None, name=None, source=None):
        if probability is None:
            return 0

        if group is None:  # Simple fact
            if identifier not in self.facts:
                if self._is_simple_probability(probability):
                    p = random.random()
                    prob = float(probability)
                    value = p < prob
                    if value:
                        result_node = self.TRUE
                        self.probability *= prob
                    else:
                        result_node = self.FALSE
                        self.probability *= (1 - prob)
                else:
                    value, prob = self.sample_value(probability)
                    self.probability *= prob
                    result_node = self.add_value(value)
                self.facts[identifier] = result_node
                return result_node
            else:
                return self.facts[identifier]
        else:
            # choice = identifier[-1]
            origin = identifier[:-1]
            if identifier not in self.facts:
                if self._is_simple_probability(probability):
                    p = float(probability)
                    if origin in self.groups:
                        r = self.groups[origin]  # remaining probability in the group
                    else:
                        r = 1.0

                    if r is None or r < 1e-8:
                        # r is too small or another choice was made for this origin
                        value = False
                    else:
                        value = (random.random() <= p / r)
                    if value:
                        self.probability *= p
                        self.groups[origin] = None   # Other choices in group are not allowed
                    elif r is not None:
                        self.groups[origin] = r - p   # Adjust remaining probability
                    if value:
                        result_node = self.TRUE
                    else:
                        result_node = self.FALSE
                else:
                    value, prob = self.sample_value(probability)
                    self.probability *= prob
                    result_node = self.add_value(value)
                self.facts[identifier] = result_node
                return result_node
            else:
                return self.facts[identifier]

    def compute_probability(self):
        for k, p in self.groups.items():
            if p is not None:
                self.probability *= p
        self.groups = {}

    def add_and(self, content, **kwdargs):
        i = 0
        for c in content:
            if c is not None and c != 0:
                i += 1
            if i > 1:
                raise ValueError("Can't combine sampled predicates.")
        return LogicFormula.add_and(self, content)

    def add_or(self, content, **kwd):
        i = 0
        for c in content:
            if c is not None and c != 0:
                i += 1
            if i > 1:
                raise ValueError("Bodies for same head should be mutually exclusive.")
        return LogicFormula.add_or(self, content, **kwd)

    def add_not(self, child):
        if child == self.TRUE:
            return self.FALSE
        elif child == self.FALSE:
            return self.TRUE
        else:
            raise ValueError("Can't negate a sampled predicate.")
        
    def is_probabilistic(self, key):
        """Indicates whether the given node is probabilistic."""
        return False

    # noinspection PyUnusedLocal
    def to_string(self, db, with_facts=False, with_probability=False, oneline=False,
                  as_evidence=False, strip_tag=False, **extra):
        self.compute_probability()

        if as_evidence:
            base = 'evidence(%s).'
        else:
            base = '%s.'

        lines = []
        for k, v in self.queries():
            if k.functor.startswith('hidden_'):
                continue
            if strip_tag:
                k = k.args[0]
            if v is not None:
                val = self.get_value(v)
                if val is None:
                    lines.append(base % (str(k)))
                else:
                    if not as_evidence:
                        lines.append('%s = %s.' % (str(k), val))
            elif as_evidence:
                lines.append(base % ('\+' + str(k)))
        if with_facts:
            for k, v in self.facts.items():
                if v == 0:
                    lines.append(base % str(translate(db, k)))
                elif v is None:
                    lines.append(base % ('\+' + str(translate(db, k))))

        if oneline:
            sep = ' '
        else:
            sep = '\n'
        lines = list(set(lines))
        if with_probability:
            lines.append('%% Probability: %.8g' % self.probability)
        return sep.join(lines)

    def to_dict(self):
        result = {}
        for k, v in self.queries():
            if v is not None:
                val = self.get_value(v)
                if val is None:
                    result[k] = True
                else:
                    result[k] = val
            else:
                result[k] = False
        return result


def translate(db, atom_id):
    if type(atom_id) == tuple:
        atom_id, args, choice = atom_id
        return Term('ad_%s_%s' % (atom_id, choice), *args)
    else:
        node = db.get_node(atom_id)
        return Term(node.functor, *node.args)


def builtin_sample(term, result, target=None, engine=None, callback=None, **kwdargs):
    # TODO wrong error message when call argument is wrong
    check_mode((term, result), ['cv'], functor='sample')
    # Find the define node for the given query term.
    term_call = term.with_args(*term.args)
    results = engine.call(term_call, subcall=True, target=target, **kwdargs)
    actions = []
    n = len(term.args)
    for res, node in results:
        res1 = res[:n]
        res_pass = (term.with_args(*res1), target.get_value(node))
        actions += callback.notifyResult(res_pass, 0, False)
    actions += callback.notifyComplete()
    return True, actions


def builtin_previous(term, default, engine=None, target=None, callback=None, **kwdargs):
    # retrieve term from previous sample, default action if no previous sample

    if engine.previous_result is None:
        results = engine.call(default, subcall=True, target=target, **kwdargs)
        actions = []
        for res, node in results:
            try:
                source_values = {}
                result_default = unify_value(default, default(*res), source_values)
                result_term = term.apply(source_values)
                actions += callback.notifyResult((result_term, result_default), node, False)
            except UnifyError:
                pass
        actions += callback.notifyComplete()

    else:
        actions = []
        for n, i in engine.previous_result.queries():
            try:
                source_values = {}
                result_term = unify_value(term, n, source_values)
                result_default = default.apply(source_values)
                if i == 0 or i is None:
                    v = i
                else:
                    v = target.add_value(engine.previous_result.get_value(i))
                actions += callback.notifyResult((result_term, result_default), v, False)
            except UnifyError:
                pass
        actions += callback.notifyComplete()
    return True, actions


def ground(engine, db, target):
    db = engine.prepare(db)
    # Load queries: use argument if available, otherwise load from database.
    queries = [q[0] for q in engine.query(db, Term('query', None))]
    evidence = engine.query(db, Term('evidence', None, None))
    evidence += engine.query(db, Term('evidence', None))

    for query in queries:
        if not isinstance(query, Term):
            raise GroundingError('Invalid query')   # TODO can we add a location?

    for ev in evidence:
        if not isinstance(ev[0], Term):
            raise GroundingError('Invalid evidence')   # TODO can we add a location?

    # Ground queries
    queries = [(target.LABEL_QUERY, q) for q in queries]
    engine.ground_queries(db, target, queries)
    return target


def init_engine(**kwdargs):
    engine = DefaultEngine(**kwdargs)
    engine.add_builtin('sample', 2, builtin_sample)
    engine.add_builtin('value', 2, builtin_sample)
    engine.add_builtin('previous', 2, builtin_previous)
    engine.previous_result = None
    return engine


def init_db(engine, model, propagate_evidence=False):
    db = engine.prepare(model)

    if propagate_evidence:
        evidence = engine.query(db, Term('evidence', None, None))
        evidence += engine.query(db, Term('evidence', None))

        ev_target = LogicFormula()
        engine.ground_evidence(db, ev_target, evidence)
        ev_target.lookup_evidence = {}
        ev_nodes = [node for name, node in ev_target.evidence()
                    if node != 0 and node is not None]
        ev_target.propagate(ev_nodes, ev_target.lookup_evidence)

        evidence_facts = []
        for index, value in ev_target.lookup_evidence.items():
            node = ev_target.get_node(index)
            if ev_target.is_true(value):
                evidence_facts.append((node[0], 1.0) + node[2:])
            elif ev_target.is_false(value):
                evidence_facts.append((node[0], 0.0) + node[2:])
    else:
        evidence_facts = []
        ev_target = None

    return db, evidence_facts, ev_target


def sample(model, n=1, format='str', propagate_evidence=False, distributions=None, **kwdargs):
    engine = init_engine(**kwdargs)
    db, evidence, ev_target = init_db(engine, model, propagate_evidence)
    i = 0
    r = 0

    while i < n:
        target = SampledFormula()
        if distributions is not None:
            target.distributions.update(distributions)

        for ev_fact in evidence:
            target.add_atom(*ev_fact)

        engine.functions = FunctionStore(target=target, database=db, engine=engine)
        result = ground(engine, db, target=target)
        if verify_evidence(engine, db, ev_target, target):
            if format == 'str':
                yield result.to_string(db, **kwdargs)
            else:
                yield result.to_dict()
            i += 1
        else:
            r += 1
        engine.previous_result = result
    if r:
        logging.getLogger('problog_sample').info('Rejected samples: %s' % r)


def verify_evidence(engine, db, ev_target, q_target):

    if ev_target is None:
        evidence = engine.query(db, Term('evidence', None, None))
        evidence += engine.query(db, Term('evidence', None))

        engine.ground_evidence(db, q_target, evidence)
        for name, node in q_target.evidence():
            if q_target.is_false(node):
                return False
        return True
    else:
        weights = {}
        for k, v in q_target.facts.items():
            p = 1.0 if q_target.is_true(v) else 0.0
            i = ev_target.add_atom(k, p)
            weights[i] = p
            weights[-i] = 1.0 - p

        for k, n, t in ev_target:
            if t == 'atom' and n.group is not None and not k in weights:
                if q_target.groups.get(n.group, 1.0) is None:
                    p = 0.0
                else:
                    p = 1.0
                weights[k] = p
                weights[-k] = 1.0 - p

        relevant_nodes = set([x for x, y in enumerate(ev_target.extract_relevant()) if y])

        update = True
        while relevant_nodes and update:
            update = False
            next_iteration = set()
            while relevant_nodes:
                i = relevant_nodes.pop()
                if i in weights:
                    w = weights[i]
                else:
                    n = ev_target.get_node(i)
                    t = type(n).__name__
                    if t == 'atom':
                        w = float(n.probability)
                        weights[i] = w
                        weights[-i] = 1.0 - w
                        update = True
                    elif t == 'disj':
                        cn = [weights[c] for c in n.children if c in weights]
                        if cn:
                            w = max(cn)
                        else:
                            w = 0.0
                        if w > 0.0:
                            weights[i] = w
                            weights[-i] = 1.0 - w
                            update = True
                        elif len(cn) != len(n.children):
                            next_iteration.add(i)
                        else:
                            weights[i] = w
                            weights[-i] = 1.0 - w
                            update = True

                    elif t == 'conj':
                        cn = [weights[c] for c in n.children if c in weights]
                        if cn:
                            w = min(cn)
                        else:
                            w = 1.0

                        if w == 0.0:
                            weights[i] = w
                            update = True
                        elif len(cn) != len(n.children):
                            next_iteration.add(i)
                        else:
                            weights[i] = w
                            update = True
            relevant_nodes = next_iteration

        for e, v in ev_target.evidence():
            if weights.get(v, 0.0) == 0.0:
                return False
        return True


# noinspection PyUnusedLocal
def estimate(model, n=0, propagate_evidence=False, **kwdargs):
    from collections import defaultdict

    engine = init_engine(**kwdargs)
    db, evidence, ev_target = init_db(engine, model, propagate_evidence)

    start_time = time.time()
    estimates = defaultdict(float)
    counts = 0.0
    r = 0
    try:
        while n == 0 or counts < n:
            target = SampledFormula()
            for ev_fact in evidence:
                target.add_atom(*ev_fact)

            result = ground(engine, db, target=target)
            if verify_evidence(engine, db, ev_target, target):
                for k, v in result.queries():
                    if v == 0:
                        estimates[k] += 1.0
                counts += 1.0
            else:
                r += 1
            engine.previous_result = result
    except KeyboardInterrupt:
        pass
    except SystemExit:
        pass

    total_time = time.time() - start_time
    rate = counts / total_time
    print ('%% Probability estimate after %d samples (%.4f samples/second):' % (counts, rate))

    if r:
        logging.getLogger('problog_sample').info('Rejected samples: %s' % r)

    for k in estimates:
        estimates[k] = estimates[k] / counts
    return estimates


def print_result(result, output=sys.stdout, oneline=False):
    success, result = result
    if success:
        # result is a sequence of samples
        first = True
        for s in result:
            if not oneline and not first:
                print ('----------------', file=output)
            first = False
            print (s, file=output)
    else:
        print (process_error(result), file=output)


def print_result_json(d, output, **kwdargs):
    """Pretty print result.

    :param d: result from run_problog
    :param output: output file
    :param precision:
    :return:
    """
    import json
    result = {}
    success, d = d
    if success:
        result['SUCCESS'] = True
        result['results'] = [[(str(k), str(v)) for k, v in dc.items()] for dc in d]
    else:
        result['SUCCESS'] = False
        result['test'] = str(type(d))
        result['err'] = process_error(d)
        result['original'] = str(d)
    print (json.dumps(result), file=output)
    return 0


def main(args, result_handler=None):
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('filename')
    parser.add_argument('-N', type=int, dest='n', default=argparse.SUPPRESS,
                        help="Number of samples.")
    parser.add_argument('--with-facts', action='store_true',
                        help="Also output choice facts (default: just queries).")
    parser.add_argument('--with-probability', action='store_true', help="Show probability.")
    parser.add_argument('--as-evidence', action='store_true', help="Output as evidence.")
    parser.add_argument('--propagate-evidence', dest='propagate_evidence',
                        default=True,
                        action='store_true', help="Enable evidence propagation")
    parser.add_argument('--dont-propagate-evidence', action='store_false',
                        dest='propagate_evidence',
                        default=True,
                        help="Disable evidence propagation")
    parser.add_argument('--oneline', action='store_true', help="Format samples on one line.")
    parser.add_argument('--estimate', action='store_true',
                        help='Estimate probability of queries from samples.')
    parser.add_argument('--timeout', '-t', type=int, default=0,
                        help="Set timeout (in seconds, default=off).")
    parser.add_argument('--output', '-o', type=str, default=None, help="Filename of output file.")
    parser.add_argument('--web', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--verbose', '-v', action='count', help='Verbose output')
    parser.add_argument('--seed', '-s', type=float, help='Random seed', default=None)
    parser.add_argument('--full-trace', action='store_true')
    parser.add_argument('--strip-tag', action='store_true', help='Strip outermost tag from output.')
    parser.add_argument('-a', '--arg', dest='args', action='append',
                        help='Pass additional arguments to the cmd_args builtin.')


    args = parser.parse_args(args)

    init_logger(args.verbose, 'problog_sample')

    if args.seed is not None:
        random.seed(args.seed)
    else:
        seed = random.random()
        logging.getLogger('problog_sample').debug('Seed: %s', seed)
        random.seed(seed)

    pl = PrologFile(args.filename)

    outf = sys.stdout
    if args.output is not None:
        outf = open(args.output, 'w')

    if args.timeout:
        start_timer(args.timeout)

    # noinspection PyUnusedLocal
    def signal_term_handler(*sigargs):
        sys.exit(143)

    signal.signal(signal.SIGTERM, signal_term_handler)

    if result_handler is not None or args.web:
        outformat = 'dict'
        if result_handler is None:
            result_handler = print_result_json
    else:
        outformat = 'str'
        result_handler = print_result
    try:
        if args.estimate:
            results = estimate(pl, **vars(args))
            print (format_dictionary(results))
        else:
            result_handler((True, sample(pl, format=outformat, **vars(args))),
                           output=outf, oneline=args.oneline)
    except Exception as err:
        trace = traceback.format_exc()
        err.trace = trace
        result_handler((False, err), output=outf)

    if args.timeout:
        stop_timer()

    if args.output is not None:
        outf.close()


if __name__ == '__main__':
    main(sys.argv[1:])
