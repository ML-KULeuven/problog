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
import os
import argparse
import itertools
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from problog.pgm.cpd import Variable, Factor, PGM


logger = logging.getLogger('be.kuleuven.cs.dtai.problog.uai2problog')


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
                break
            if '#' in line:
                line = line[:line.index('#')]
            tokens += line.strip().split()
        self.buffer = tokens[amount:]
        tokens = tokens[:amount]
        return tokens

    def get_token(self):
        tokens = self.get_tokens(1)
        if tokens is None or len(tokens) == 0:
            return None
        return tokens[0]

    def __del__(self):
        logger.debug('Closing {}'.format(self.file.name))
        if self.file is not None:
            self.file.close()


class UAIParser:

    def __init__(self, fn):
        self.fn = fn
        self.reader = None
        self.pgm = None

        self.force_bool = False
        self.detect_bool = True
        self.drop_zero = False
        self.use_neglit = False

        self.directed = True
        self.num_vars = 0
        self.num_funcs = 0
        self.dom_sizes = []
        self.domains = []
        self.func_vars = []
        self.func_values = []
        self.factor_cnt = 0

    def construct_var(self, var_num):
        rv = str(var_num)
        values = self.domains[var_num]
        self.pgm.add_var(Variable(rv, values, detect_boolean=self.detect_bool, force_boolean=self.force_bool))

    def construct_cpt(self, func_num):
        if func_num >= len(self.func_vars) or self.func_vars[func_num] is None:
            error('Variables not defined for function {}'.format(func_num), halt=True)
        variables = self.func_vars[func_num]
        var = variables[-1]
        rv = str(var)
        parents = variables[0:-1]
        if var >= len(self.domains) or self.domains[var] is None:
            error('Variable domain is not defined: {}'.format(var), halt=True)
        dom_size = self.dom_sizes[var]

        parents_str = [str(p) for p in parents]
        if len(parents) == 0:
            table = self.func_values[var]
            self.pgm.add_factor(Factor(self.pgm, rv, parents_str, table))
            return
        parent_domains = []
        for parent in parents:
            parent_domains.append(self.domains[parent])
        table = {}
        try:
            cur_func_values = self.func_values[func_num]
        except KeyError:
            error('Could not find function definition for {}'.format(func_num), halt=True)
        else:
            idx = 0
            for val_assignment in itertools.product(*parent_domains):
                table[val_assignment] = cur_func_values[idx:idx+dom_size]
                idx += dom_size

        self.pgm.add_factor(Factor(self.pgm, rv, parents_str, table))

    def construct_factor(self, func_num):
        if func_num >= len(self.func_vars) or self.func_vars[func_num] is None:
            error('Variables not defined for function {}'.format(func_num), halt=True)
        variables = self.func_vars[func_num]
        name = 'pf_{}'.format(self.factor_cnt)
        self.factor_cnt += 1
        parents = variables
        dom_size = 1

        parents_str = [str(p) for p in parents]
        assert(len(parents) != 0)
        parent_domains = []
        for parent in parents:
            parent_domains.append(self.domains[parent])
        table = {}
        try:
            cur_func_values = self.func_values[func_num]
        except KeyError:
            error('Could not find function definition for {}'.format(func_num), halt=True)
        else:
            idx = 0
            for val_assignment in itertools.product(*parent_domains):
                table[val_assignment] = cur_func_values[idx:idx + dom_size]
                idx += dom_size

        self.pgm.add_factor(Factor(self.pgm, None, parents_str, table, name=name))

    def construct_pgm(self):
        self.pgm = PGM(directed=self.directed)
        for var_num in range(self.num_vars):
            self.construct_var(var_num)
        for func_num in range(self.num_funcs):
            if self.directed:
                self.construct_cpt(func_num)
            else:
                self.construct_factor(func_num)

    def parse_header(self):
        logger.debug('Parsing header')

        # Type
        token = self.reader.get_token()
        if token == 'BAYES':
            self.directed = True
        elif token == 'MARKOV':
            self.directed = False
        else:
            self.directed = None
            error('Expected a BAYES or MARKOV network, found: ' +
                  '{} (line {})'.format(token, self.reader.linenb), halt=True)
        logger.debug('Type: {}'.format(token))

        # Number of variables
        token = self.reader.get_token()
        self.num_vars = int(token)
        logger.debug("Number of variables: {}".format(self.num_vars))

        # Domain sizes
        tokens = self.reader.get_tokens(self.num_vars)
        for size in tokens:
            size = int(size)
            self.dom_sizes.append(size)
            values = [str(d) for d in range(size)]
            self.domains.append(values)
        if len(self.dom_sizes) != self.num_vars:
            error('Expected {} domain sizes, '.format(self.num_vars) +
                  'found {} (line {})'.format(len(self.dom_sizes), self.reader.linenb), halt=True)
        logger.debug("Domain sizes: {}".format(" ".join(map(str, self.dom_sizes))))

        # Number of functions
        token = self.reader.get_token()
        self.num_funcs = int(token)
        if self.directed and self.num_funcs != self.num_vars:
            error('For BAYES we expect one function for every variables but found: ' +
                  '{} (line {})'.format(self.num_funcs, self.reader.linenb), halt=True)
        logger.debug("Number of functions: {}".format(self.num_funcs))

    def parse_graph(self):
        logger.debug('Parsing function structures')
        for num_func in range(self.num_funcs):
            logger.debug('Parsing function structure {}'.format(num_func))
            func_size = int(self.reader.get_token())
            tokens = self.reader.get_tokens(func_size)
            tokens = [int(v) for v in tokens]

            if len(tokens) != func_size:
                error('Expected {} variables, found {}\n{}'.format(func_size, len(tokens), tokens), halt=True)
            # if num_func != rv:
            #     error('Expected current variable ({}) as last variable, found {}'.format(num_func, rv), halt=True)
            # for parent in parents:
            #     if parent >= num_func:
            #         error('Found parent ({}) that is not yet defined\n{}'.format(parent, line), halt=True)
            self.func_vars.append(tokens)
            logger.debug('Parsed structure: {}'.format(" ".join(map(str, tokens))))

    def parse_functions(self):
        logger.debug('Parsing function values')
        for num_func in range(self.num_funcs):
            logger.debug('Parsing function values {}'.format(num_func))
            num_values = int(self.reader.get_token())
            exp_num_values = 1
            for var in self.func_vars[num_func]:
                exp_num_values *= self.dom_sizes[var]
            if exp_num_values != num_values:
                logger.warning(('% WARNING: Function {} says {} values but {} values ' +
                                'are expected given the domain size ' +
                                'for variable {}.').format(num_func, num_values, exp_num_values, num_func))
            tokens = self.reader.get_tokens(num_values)
            values = [float(v) for v in tokens]
            if len(values) != num_values:
                error('Expected {} values in function {}, '.format(num_values, num_func) +
                      'found {} (line {})'.format(len(values), self.reader.linenb), halt=True)
            self.func_values.append(values)

    def parse_rest(self):
        self.reader.get_token()
        # for line in reader:
        #     line = line.strip()
        #     if line != '':
        #         warning('Did not expect more lines, ignoring: {}'.format(line))

    def parse(self):
        self.reader = UAIReader(self.fn)

        self.parse_header()
        self.parse_graph()
        self.parse_functions()
        self.parse_rest()

        self.construct_pgm()
        return self.pgm

    def print_datastructures(self):
        print('Domain sizes: {}'.format(' '.join([str(s) for s in self.dom_sizes])))
        print('Function structures:\n' +
              '  {}'.format('\n  '.join([' '.join([str(pp) for pp in p]) for p in self.func_vars])))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Translate Bayesian net in UAI08 format to ProbLog')
    parser.add_argument('--verbose', '-v', action='count', help='Verbose output')
    parser.add_argument('--forcebool', action='store_true',
                        help='Force binary nodes to be represented as Boolean predicates (0=f, 1=t)')
    parser.add_argument('--nobooldetection', action='store_true', help='Do not try to detect Boolean predicates')
    parser.add_argument('--dropzero', action='store_true', help='Drop zero probabilities (if possible)')
    parser.add_argument('--useneglit', action='store_true', help='Use negative head literals')
    parser.add_argument('--allowdisjunct', action='store_true', help='Allow disjunctions in the body')
    parser.add_argument('--compress', action='store_true', help='Compress tables')
    parser.add_argument('--split', help='Comma-separated list of variable names to split on')
    parser.add_argument('--split-output', dest='splitoutput', action='store_true',
                        help='Create one output file per connected network')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--output-format', default='problog',
                        help='Output format (\'problog\', \'uai\', \'hugin\', \'xdsl\')')
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

    uai_parser = UAIParser(args.input)

    if args.forcebool:
        uai_parser.force_bool = args.forcebool
    if args.nobooldetection:
        uai_parser.detect_bool = False
    if args.dropzero:
        uai_parser.drop_zero = args.dropzero
    if args.useneglit:
        uai_parser.use_neglit = args.useneglit

    pgm = uai_parser.parse()
    if args.compress:
        pgm = pgm.compress_tables(allow_disjunct=args.allowdisjunct)
    if args.split:
        pgm = pgm.split(set(args.split.split(',')))
    if pgm is None:
        error('Could not build PGM structure', halt=True)
    if args.splitoutput:
        pgms = pgm.to_connected_parts()
    else:
        pgms = [pgm]

    for pgm_i, pgm in enumerate(pgms):
        if args.output:
            if len(pgms) == 1:
                fn = args.output
            else:
                fn_base, fn_ext = os.path.splitext(args.output)
                fn = fn_base + '.' + str(pgm_i) + fn_ext
            ofile = open(fn, 'w')
        else:
            ofile = sys.stdout
        try:
            if args.output_format in ["uai", "uai08"]:
                print(pgm.to_uai08(), file=ofile)
            elif args.output_format == "hugin":
                print(pgm.to_hugin_net(), file=ofile)
            elif args.output_format in ["smile", "xdsl"]:
                print(pgm.to_xdsl(), file=ofile)
            elif args.output_format in ["graphiz", "dot"]:
                print(pgm.to_graphviz(), file=ofile)
            else:
                print(pgm.to_problog(drop_zero=uai_parser.drop_zero, use_neglit=uai_parser.use_neglit), file=ofile)
        finally:
            if args.output:
                ofile.close()

if __name__ == "__main__":
    sys.exit(main())
