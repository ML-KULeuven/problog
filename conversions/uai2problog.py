#!/usr/bin/env python3
# encoding: utf-8
"""
uai2problog.py

Created by Wannes Meert on 31-01-2016.
Copyright (c) 2016 KU Leuven. All rights reserved.
"""

import sys
import argparse
from problog.bn.cpd import CPT, PGM
import itertools

verbose = 0
force_bool = False
drop_zero = False
use_neglit = False

num_vars = 0
num_funcs = 0
dom_sizes = []
domains = []
var_parents = []
func_values = []


def info(string):
    if verbose >= 1:
        print('INFO: '+string)

def debug(string):
    if verbose >= 2:
        print('DEBUG: '+string)

def warning(string):
    if verbose >= 0:
        print('WARNING: '+string)

def error(string, halt=False):
    print('ERROR: '+string)
    if halt:
        sys.exit(1)

def construct_cpt(var):
    global domains
    rv = 'v{}'.format(var)
    if force_bool and dom_sizes[var] == 2:
        values = ['f', 't']
    else:
        values = [str(d) for d in range(dom_sizes[var])]
    domains.append(values)
    parents = ['v{}'.format(p) for p in var_parents[var]]
    if len(parents) == 0:
        table = func_values[var]
        return CPT(rv, values, parents, table)
    parent_domains = []
    for parent in var_parents[var]:
        parent_domains.append(domains[parent])
    idx = 0
    try:
        cur_func_values = func_values[var]
    except:
        error('Could not find function definition for {}'.format(var), halt=True)
    dom_size = dom_sizes[var]
    table = {}
    for val_assignment in itertools.product(*parent_domains):
        table[val_assignment] = cur_func_values[idx:idx+dom_size]
        idx += dom_size
    return CPT(rv, values, parents, table)

def construct_pgm():
    pgm = PGM()
    for var_num in range(num_vars):
        cpt = construct_cpt(var_num)
        debug(str(cpt))
        pgm.add(cpt)
    return pgm

def parse_header(ifile):
    debug('Parsing header')
    # Type
    line = ifile.readline().strip()
    if line != 'BAYES':
        error('Expected a BAYES network, found: {}'.format(line), halt=True)
    # Number of variables
    global num_vars
    line = ifile.readline().strip()
    num_vars = int(line)
    # Domain sizes
    global dom_sizes
    line = ifile.readline().strip()
    dom_sizes = [int(size) for size in line.split()]
    if len(dom_sizes) != num_vars:
        error('Expected {} domain sizes, found {}'.format(num_vars, len(dom_sizes)), halt=True)
    # Number of functions
    line = ifile.readline().strip()
    global num_funcs
    num_funcs = int(line)
    if num_funcs != num_vars:
        error('For BAYES we expect one function for every variables but found: {}'.format(num_funcs), halt=True)

def parse_graph(ifile):
    global var_parents
    debug('Parsing function structures')
    for num_var in range(num_funcs):
        debug('Parsing function structure {}'.format(num_var))
        line = ifile.readline().strip()
        if line == '':
            error('Did not expect empty line for function definition {}'.format(num_var), halt=True)
        line = [int(v) for v in line.split()]
        func_size = line[0]
        parents = line[1:-1]
        rv = line[-1]
        if len(parents) != func_size-1:
            error('Expected {} parents, found {}\n{}'.format(func_size-1, len(parents), line), halt=True)
        if num_var != rv:
            error('Expected current variable ({}) as last variable, found {}'.format(num_var, rv), halt=True)
        for parent in parents:
            if parent >= num_var:
                error('Found parent ({}) that is not yet defined\n{}'.format(parent, line), halt=True)
        var_parents.append(parents)

def parse_functions(ifile):
    global func_values
    debug('Parsing function values')
    line = ifile.readline().strip()
    if line != '':
        error('Expected empty line, found: {}'.format(line), halt=True)
    for num_func in range(num_funcs):
        debug('Parsing function values {}'.format(num_func))
        line = ifile.readline().strip()
        while line == '':
            line = ifile.readline().strip()
        num_values = int(line)
        values = []
        while len(values) < num_values:
            line = [float(v) for v in ifile.readline().strip().split()]
            values += line
        if len(values) != num_values:
            error('Expected {} values for function {}, found {}'.format(num_values, num_func, len(values)), halt=True)
        func_values.append(values)

def parse_rest(ifile):
    for line in ifile:
        line = line.strip()
        if line != '':
            warning('Did not expect more lines, ignoring: {}'.format(line))

def parse(ifile):
    parse_header(ifile)
    parse_graph(ifile)
    parse_functions(ifile)
    parse_rest(ifile)

def print_datastructures():
    print('Domain sizes: {}'.format(' '.join([str(s) for s in dom_sizes])))
    print('Function structures:\n  {}'.format('\n  '.join([' '.join([str(pp) for pp in p]) for p in var_parents])))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Translate Bayesian net in UAI08 format to ProbLog')
    parser.add_argument('--verbose', '-v', action='count', help='Verbose output')
    parser.add_argument('--forcebool', action='store_true', help='Force binary nodes to be represented as boolean predicates (0=f, 1=t)')
    parser.add_argument('--dropzero', action='store_true', help='Drop zero probabilities (if possible)')
    parser.add_argument('--useneglit', action='store_true', help='Use negative head literals')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('input', help='Input UAI08 file')
    args = parser.parse_args(argv)

    global verbose
    if args.verbose is not None:
        verbose = args.verbose
    global force_bool
    if args.forcebool:
        force_bool = args.forcebool
    global drop_zero
    if args.dropzero:
        drop_zero = args.dropzero
    global use_neglit
    if args.useneglit:
        use_neglit = args.useneglit

    with open(args.input, 'r') as ifile:
        parse(ifile)
    pgm = construct_pgm()
    if pgm is None:
        error('Could not build PGM structure', halt=True)
    if args.output is None:
        ofile = sys.stdout
    else:
        ofile = open(args.output, 'w')
    print(pgm.to_problog(drop_zero=drop_zero, use_neglit=use_neglit), file=ofile)


if __name__ == "__main__":
    sys.exit(main())

