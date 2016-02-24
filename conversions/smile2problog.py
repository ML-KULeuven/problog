#!/usr/bin/env python3
# encoding: utf-8
"""
hugin2problog.py

Created by Wannes Meert on 23-02-2016.
Copyright (c) 2016 KU Leuven. All rights reserved.
"""

import sys
import argparse
from problog.bn.cpd import CPT, PGM
import itertools
import logging
import re
import xml.etree.ElementTree as ET

force_bool = False
drop_zero = False
use_neglit = False
no_bool_detection = False

domains = {}
potentials = []
cpds = []

logger = logging.getLogger('problog.smile2problog')

def info(*args, **kwargs):
    logger.info(*args, **kwargs)

def debug(*args, **kwargs):
    logger.debug(*args, **kwargs)

def warning(*args, **kwargs):
    logger.warning(*args, **kwargs)

def error(*args, **kwargs):
    if 'halt' in kwargs:
        halt = kwargs['halt']
        del kwargs['halt']
    logger.error(*args, **kwargs)
    if halt:
        sys.exit(1)

## PARSER

def parse(ifile):
    tree = ET.parse(ifile)
    root = tree.getroot()

    parseDomains(root)
    for cpt in root.find("nodes").findall('cpt'):
        parseCPT(cpt)

def parseDomains(root):
    for cpt in root.find("nodes").findall('cpt'):
        rv = cpt.get('id')
        states = cpt.findall('state')
        values = [state.get('id') for state in states]
        domains[rv] = values

def parseCPT(cpt):
    global cpds
    detect_boolean = not no_bool_detection
    rv = cpt.get('id')
    if rv not in domains:
        error('Domain for {} not defined.'.format(rv), halt=True)
    values = domains[rv]
    parents = cpt.find('parents')
    if parents is None:
        parents = []
    else:
        parents = parents.text.split()
    parameters = [float(p) for p in cpt.find('probabilities').text.split()]
    if len(parents) == 0:
        table = parameters
        cpds.append(CPT(rv, values, parents, table, detect_boolean=detect_boolean))
        return
    parent_domains = []
    for parent in parents:
        parent_domains.append(domains[parent])
    dom_size = len(values)
    table = {}
    idx = 0
    for val_assignment in itertools.product(*parent_domains):
        table[val_assignment] = parameters[idx:idx+dom_size]
        idx += dom_size
    cpds.append(CPT(rv, values, parents, table, detect_boolean=detect_boolean))


def construct_pgm():
    return PGM(cpds=cpds)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Translate Bayesian net in Hugin format format to ProbLog')
    parser.add_argument('--verbose', '-v', action='count', help='Verbose output')
    parser.add_argument('--nobooldetection', action='store_true',
                        help='Do not try to infer if a node is Boolean (true/false, yes/no, ...)')
    parser.add_argument('--forcebool', action='store_true',
                        help='Force all binary nodes to be represented as boolean predicates (0=f, 1=t)')
    parser.add_argument('--dropzero', action='store_true', help='Drop zero probabilities (if possible)')
    parser.add_argument('--useneglit', action='store_true', help='Use negative head literals')
    parser.add_argument('--valueinatomname', action='store_false',
                        help='Add value to atom name instead as a term (this removes invalid characters, be careful \
                              that clean values do not overlap)')
    parser.add_argument('--compress', action='store_true', help='Compress tables')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('input', help='Input Hugin file')
    args = parser.parse_args(argv)

    if args.verbose is None:
        logger.setLevel(logging.WARNING)
    elif args.verbose == 1:
        logger.setLevel(logging.INFO)
    elif args.verbose >= 2:
        logger.setLevel(logging.DEBUG)

    global no_bool_detection
    if args.nobooldetection:
        no_bool_detection = args.nobooldetection
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
    if args.compress:
        pgm = pgm.compress_tables()
    if pgm is None:
        error('Could not build PGM structure', halt=True)

    ofile = sys.stdout
    if args.output is not None:
        ofile = open(args.output, 'w')
    print(pgm.to_problog(drop_zero=drop_zero, use_neglit=use_neglit, value_as_term=args.valueinatomname), file=ofile)


if __name__ == "__main__":
    sys.exit(main())

