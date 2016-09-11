#!/usr/bin/env python3
# encoding: utf-8
"""
uai2problog.py

http://graphmod.ics.uci.edu/uai08/FileFormat

Created by Wannes Meert on 31-01-2016.
Copyright (c) 2016 KU Leuven. All rights reserved.
"""
from __future__ import print_function

import sys
import argparse
from problog.pgm.cpd import Variable, Factor, PGM
import itertools
import logging

force_bool = False
detect_bool = True
drop_zero = False
use_neglit = False

directed = True
num_vars = 0
num_funcs = 0
dom_sizes = []
domains = []
func_vars = []
func_values = []
factor_cnt = 0


logger = logging.getLogger('problog.uai2problog')


def error(*args, **kwargs):
    halt = False
    if 'halt' in kwargs:
        halt = kwargs['halt']
        del kwargs['halt']
    logger.error(*args, **kwargs)
    if halt:
        sys.exit(1)


class UAIReader:
    def __init__(self, fn):
        logger.debug('Opening {}'.format(fn))
        self.file = open(fn, 'r')
        self.linenb = 0
        self.buffer = []

    def get_tokens(self, amount):
        if amount < len(self.buffer):
            tokens = self.buffer[:amount]
            self.buffer = self.buffer[amount:]
            return tokens
        tokens = self.buffer
        while len(tokens) < amount:
            line = self.file.readline()
            self.linenb += 1
            if line == '':
                return None
            if '#' in line:
                line = line[:line.index('#')]
            tokens += line.strip().split()
        self.buffer = tokens[amount:]
        tokens = tokens[:amount]
        return tokens

    def get_token(self):
        tokens = self.get_tokens(1)
        if tokens is None:
            return None
        return tokens[0]

    def __del__(self):
        logger.debug('Closing {}'.format(self.file.name))
        if self.file is not None:
            self.file.close()


def construct_var(pgm, var_num):
    rv = 'v{}'.format(var_num)
    values = domains[var_num]
    pgm.add_var(Variable(rv, values, detect_boolean=detect_bool, force_boolean=force_bool))


def construct_cpt(pgm, func_num):
    global factor_cnt
    global domains
    if func_num >= len(func_vars) or func_vars[func_num] is None:
        error('Variables not defined for function {}'.format(func_num), halt=True)
    variables = func_vars[func_num]
    var = variables[-1]
    rv = 'v{}'.format(var)
    parents = variables[0:-1]
    if var >= len(domains) or domains[var] is None:
        error('Variable domain is not defined: {}'.format(var), halt=True)
    dom_size = dom_sizes[var]

    parents_str = ['v{}'.format(p) for p in parents]
    if len(parents) == 0:
        table = func_values[var]
        pgm.add_factor(Factor(pgm, rv, parents_str, table))
        return
    parent_domains = []
    for parent in parents:
        parent_domains.append(domains[parent])
    idx = 0
    try:
        cur_func_values = func_values[func_num]
    except:
        error('Could not find function definition for {}'.format(func_num), halt=True)
    table = {}
    for val_assignment in itertools.product(*parent_domains):
        table[val_assignment] = cur_func_values[idx:idx+dom_size]
        idx += dom_size

    pgm.add_factor(Factor(pgm, rv, parents_str, table))


def construct_factor(pgm, func_num):
    global factor_cnt
    global domains
    if func_num >= len(func_vars) or func_vars[func_num] is None:
        error('Variables not defined for function {}'.format(func_num), halt=True)
    variables = func_vars[func_num]
    name = 'pf_{}'.format(factor_cnt)
    factor_cnt += 1
    parents = variables
    dom_size = 1

    parents_str = ['v{}'.format(p) for p in parents]
    assert(len(parents) != 0)
    parent_domains = []
    for parent in parents:
        parent_domains.append(domains[parent])
    idx = 0
    table = {}
    try:
        cur_func_values = func_values[func_num]
        for val_assignment in itertools.product(*parent_domains):
            table[val_assignment] = cur_func_values[idx:idx + dom_size]
            idx += dom_size
    except:
        error('Could not find function definition for {}'.format(func_num), halt=True)

    pgm.add_factor(Factor(pgm, None, parents_str, table, name=name))


def construct_pgm():
    pgm = PGM(directed=directed)
    for var_num in range(num_vars):
        construct_var(pgm, var_num)
    for func_num in range(num_funcs):
        if directed:
            construct_cpt(pgm, func_num)
        else:
            construct_factor(pgm, func_num)
    return pgm


def parse_header(reader):
    logger.debug('Parsing header')
    global var_parents

    # Type
    token = reader.get_token()
    global directed
    if token == 'BAYES':
        directed = True
    elif token == 'MARKOV':
        directed = False
    else:
        directed = None
        error('Expected a BAYES or MARKOV network, found: {}'.format(token), halt=True)
    logger.debug('Type: {}'.format(token))

    # Number of variables
    global num_vars
    token = reader.get_token()
    num_vars = int(token)
    var_parents = [None]*num_vars
    logger.debug("Number of variables: {}".format(num_vars))

    # Domain sizes
    global dom_sizes
    global domains
    tokens = reader.get_tokens(num_vars)
    dom_size = []
    for size in tokens:
        size = int(size)
        dom_sizes.append(size)
        values = [str(d) for d in range(size)]
        domains.append(values)
    if len(dom_sizes) != num_vars:
        error('Expected {} domain sizes, found {}'.format(num_vars, len(dom_sizes)), halt=True)
    logger.debug("Domain sizes: {}".format(" ".join(map(str,dom_sizes))))

    # Number of functions
    token = reader.get_token()
    global num_funcs
    num_funcs = int(token)
    if directed and num_funcs != num_vars:
        error('For BAYES we expect one function for every variables but found: {}'.format(num_funcs), halt=True)
    logger.debug("Number of functions: {}".format(num_funcs))


def parse_graph(reader):
    global func_vars
    logger.debug('Parsing function structures')
    for num_func in range(num_funcs):
        logger.debug('Parsing function structure {}'.format(num_func))
        func_size = int(reader.get_token())
        tokens = reader.get_tokens(func_size)
        tokens = [int(v) for v in tokens]

        if len(tokens) != func_size:
            error('Expected {} variables, found {}\n{}'.format(func_size, len(tokens), tokens), halt=True)
        # if num_func != rv:
            # error('Expected current variable ({}) as last variable, found {}'.format(num_func, rv), halt=True)
        # for parent in parents:
            # if parent >= num_func:
                # error('Found parent ({}) that is not yet defined\n{}'.format(parent, line), halt=True)
        func_vars.append(tokens)
        logger.debug('Parsed structure: {}'.format(" ".join(map(str,tokens))))


def parse_functions(reader):
    global func_values
    logger.debug('Parsing function values')
    for num_func in range(num_funcs):
        logger.debug('Parsing function values {}'.format(num_func))
        num_values = int(reader.get_token())
        exp_num_values = 1
        for var in func_vars[num_func]:
            exp_num_values *= dom_sizes[var]
        if exp_num_values != num_values:
            logger.warning('WARNING: Function {} says {} values but {} is expected.'.format(num_func, num_values, exp_num_values))
        tokens = reader.get_tokens(num_values)
        values = [float(v) for v in tokens]
        func_values.append(values)


def parse_rest(reader):
    token = reader.get_token()
    # for line in reader:
    #     line = line.strip()
    #     if line != '':
    #         warning('Did not expect more lines, ignoring: {}'.format(line))


def parse(reader):
    parse_header(reader)
    parse_graph(reader)
    parse_functions(reader)
    parse_rest(reader)


def print_datastructures():
    print('Domain sizes: {}'.format(' '.join([str(s) for s in dom_sizes])))
    print('Function structures:\n  {}'.format('\n  '.join([' '.join([str(pp) for pp in p]) for p in func_vars])))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Translate Bayesian net in UAI08 format to ProbLog')
    parser.add_argument('--verbose', '-v', action='count', help='Verbose output')
    parser.add_argument('--forcebool', action='store_true', help='Force binary nodes to be represented as Boolean predicates (0=f, 1=t)')
    parser.add_argument('--nodetectbool', action='store_true', help='Do not try to detect Boolean predicates')
    parser.add_argument('--dropzero', action='store_true', help='Drop zero probabilities (if possible)')
    parser.add_argument('--useneglit', action='store_true', help='Use negative head literals')
    parser.add_argument('--allowdisjunct', action='store_true', help='Allow disjunctions in the body')
    parser.add_argument('--compress', action='store_true', help='Compress tables')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('input', help='Input UAI08 file')
    args = parser.parse_args(argv)

    ch = logging.StreamHandler()
    if args.verbose is None:
        logger.setLevel(logging.WARNING)
        ch.setLevel(logging.WARNING)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
        ch.setLevel(logging.INFO)
    elif args.verbose >= 2:
        logger.setLevel(logging.DEBUG)
        ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    global verbose
    if args.verbose is not None:
        verbose = args.verbose
    global force_bool
    if args.forcebool:
        force_bool = args.forcebool
    global detect_bool
    if args.nodetectbool:
        detect_bool = False
    global drop_zero
    if args.dropzero:
        drop_zero = args.dropzero
    global use_neglit
    if args.useneglit:
        use_neglit = args.useneglit

    reader = UAIReader(args.input)
    parse(reader)
    pgm = construct_pgm()
    if args.compress:
        pgm = pgm.compress_tables(allow_disjunct=args.allowdisjunct)
    if pgm is None:
        error('Could not build PGM structure', halt=True)

    ofile = sys.stdout
    if args.output is not None:
        ofile = open(args.output, 'w')
    print(pgm.to_problog(drop_zero=drop_zero, use_neglit=use_neglit), file=ofile)


if __name__ == "__main__":
    sys.exit(main())

