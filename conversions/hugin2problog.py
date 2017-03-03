#!/usr/bin/env python3
# encoding: utf-8
"""
hugin2problog.py

Created by Wannes Meert on 23-02-2016.
Copyright (c) 2016 KU Leuven. All rights reserved.
"""
from __future__ import print_function

import sys
import os
import argparse
import itertools
import logging
import re
import time

from pyparsing import Word, nums, ParseException, alphanums, \
                      OneOrMore, Or, Optional, dblQuotedString, Regex, \
                      Forward, ZeroOrMore, Suppress, removeQuotes, Group, ParserElement

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from problog.pgm.cpd import Variable, Factor, PGM

ParserElement.enablePackrat()

force_bool = False
drop_zero = False
use_neglit = False
detect_bool = True

domains = {}
potentials = []
pgm = PGM()

logger = logging.getLogger('be.kuleuven.cs.dtai.problog.hugin2problog')


def info(*args, **kwargs):
    logger.info(*args, **kwargs)


def debug(*args, **kwargs):
    logger.debug(*args, **kwargs)


def warning(*args, **kwargs):
    logger.warning(*args, **kwargs)


def error(*args, **kwargs):
    halt = False
    if 'halt' in kwargs:
        halt = kwargs['halt']
        del kwargs['halt']
    logger.error(*args, **kwargs)
    if halt:
        sys.exit(1)


# PARSER

re_comments = re.compile(r"""%.*?[\n\r]""")


def rm_comments(string):
    return re_comments.sub("\n", string)

S = Suppress

p_optval = Or([dblQuotedString, S("(") + OneOrMore(Word(nums)) + S(")")])
p_option = S(Group(Word(alphanums+"_") + S("=") + Group(p_optval) + S(";")))
p_net = S(Word("net") + "{" + ZeroOrMore(p_option) + "}")
p_var = Word(alphanums+"_")
p_val = dblQuotedString.setParseAction(removeQuotes)
p_states = Group(Word("states") + S("=") + S("(") + Group(OneOrMore(p_val)) + S(")") + S(";"))
p_node = S(Word("node")) + p_var + S("{") + Group(ZeroOrMore(Or([p_states, p_option]))) + S("}")
p_par = Regex(r'\d+(\.\d*)?([eE]\d+)?')
p_parlist = Forward()
p_parlist << S("(") + Or([OneOrMore(p_par), OneOrMore(p_parlist)]) + S(")")
p_data = S(Word("data")) + S("=") + Group(p_parlist) + S(";")
p_potential = S(Word("potential")) + S("(") + p_var + Group(Optional(S("|") + OneOrMore(p_var))) + S(")") + S("{") + \
              p_data + S("}")

parser = OneOrMore(Or([p_net, p_node, p_potential]))


def parse_option(s, l, t):
    return None


def parse_node(s, l, t):
    global domains
    # print(t)
    rv = t[0]
    for key, val in t[1]:
        if key == 'states':
            domains[rv] = val
            pgm.add_var(Variable(rv, val, detect_boolean=detect_bool, force_boolean=force_bool))


def parse_potential(s, l, t):
    # print(t)
    rv = t[0]
    if rv not in domains:
        error('Domain for {} not defined.'.format(rv), halt=True)
    values = domains[rv]
    parents = t[1]
    parameters = t[2]
    if len(parents) == 0:
        table = list([float(p) for p in parameters])
        pgm.add_factor(Factor(pgm, rv, parents, table))
        return
    parent_domains = []
    for parent in parents:
        parent_domains.append(domains[parent])
    dom_size = len(values)
    table = {}
    idx = 0
    for val_assignment in itertools.product(*parent_domains):
        table[val_assignment] = [float(p) for p in parameters[idx:idx+dom_size]]
        idx += dom_size
    pgm.add_factor(Factor(pgm, rv, parents, table))

p_option.setParseAction(parse_option)
p_node.setParseAction(parse_node)
p_potential.setParseAction(parse_potential)


# Test


def tests():
    # test('net {}')
    # test('node a { states = ( x y );}')
    # test('node a { states = ( "x" "y" );}')
    # test('potential (a) { data = (0.5 0.5); }')
    # test('potential (a | b c) { data = (0.5 0.5); }')
    # test('potential (a) { data = ((0.5 0.5)); }')
    # test('potential (a) { data = ((0.5 0.5)(0.5 0.5)); }')
    # test('potential (a) { data = ((1 0)(0 1)); }')
    # test('potential (a) { data = ((1 0)(0 1)); %test\n}')
    # test('net { val = "x"; }')
    # test('net { val = (0 1); }')
    # test('net { val_x = "x"; }')
    # test('node a { label = ""; }')
    # test('node a { label = ""; states = ("1" "2"); }')
    pass


def test(string):
    try:
        parse(string)
        # print('{} -> {}\n'.format(string,result))
    except ParseException as err:
        # print(string)
        print(err)
        print('\n')


# Processing


def parse(text):
    text = rm_comments(text)
    result = None
    try:
        result = parser.parseString(text, parseAll=True)
    except ParseException as err:
        print(err)
    return result


def main(argv=None):
    description = 'Translate Bayesian net in Hugin .net/.hugin format format to ProbLog'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
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
    parser.add_argument('--split', help='Comma-separated list of variable names to split on')
    parser.add_argument('--output', '-o', help='Output file')
    parser.add_argument('--output-format', default='problog',
                        help='Output format (\'problog\', \'uai\', \'hugin\', \'xdsl\')')
    parser.add_argument('input', help='Input Hugin file')
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    global detect_bool
    if args.nobooldetection:
        detect_bool = False
    global force_bool
    if args.forcebool:
        force_bool = args.forcebool
    global drop_zero
    if args.dropzero:
        drop_zero = args.dropzero
    global use_neglit
    if args.useneglit:
        use_neglit = args.useneglit

    text = None
    global pgm
    logger.info("Start parsing ...")
    ts1 = time.clock()
    with open(args.input, 'r') as ifile:
        text = ifile.read()
    parse(text)
    ts2 = time.clock()
    logger.info("Parsing took {:.3f} sec".format(ts2 - ts1))
    ts1 = ts2
    if args.compress:
        logger.info("Start compressing ...")
        pgm = pgm.compress_tables()
        ts2 = time.clock()
        logger.info("Compressing tables took {:.3f} sec".format(ts2 - ts1))
        ts1 = ts2
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
                print(pgm.to_problog(drop_zero=drop_zero, use_neglit=use_neglit, value_as_term=args.valueinatomname),
                      file=ofile)
        finally:
            if args.output:
                ofile.close()


if __name__ == "__main__":
    sys.exit(main())

