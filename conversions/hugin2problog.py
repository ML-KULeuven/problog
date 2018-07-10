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

from bn2problog import BNParser

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from problog.pgm.cpd import Variable, Factor, PGM

ParserElement.enablePackrat()
logger = logging.getLogger('be.kuleuven.cs.dtai.problog.bn2problog')


class HuginParser(BNParser):

    re_comments = re.compile(r"""%.*?[\n\r]""")

    @staticmethod
    def rm_comments(string):
        return HuginParser.re_comments.sub("\n", string)

    def __init__(self, args):
        super(HuginParser, self).__init__(args)

        self.domains = {}
        self.potentials = []

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

        p_option.setParseAction(self.parse_option)
        p_node.setParseAction(self.parse_node)
        p_potential.setParseAction(self.parse_potential)

        self.parser = OneOrMore(Or([p_net, p_node, p_potential]))

    def parse_option(self, s, l, t):
        return None

    def parse_node(self, s, l, t):
        # print(t)
        rv = t[0]
        for key, val in t[1]:
            if key == 'states':
                self.domains[rv] = val
                self.pgm.add_var(Variable(rv, val, detect_boolean=self.detect_bool, force_boolean=self.force_bool))

    def parse_potential(self, s, l, t):
        # print(t)
        rv = t[0]
        if rv not in self.domains:
            logger.error('Domain for {} not defined.'.format(rv), halt=True)
            sys.exit(1)
        values = self.domains[rv]
        parents = t[1]
        parameters = t[2]
        if len(parents) == 0:
            table = list([float(p) for p in parameters])
            self.pgm.add_factor(Factor(self.pgm, rv, parents, table))
            return
        parent_domains = []
        for parent in parents:
            parent_domains.append(self.domains[parent])
        dom_size = len(values)
        table = {}
        idx = 0
        for val_assignment in itertools.product(*parent_domains):
            table[val_assignment] = [float(p) for p in parameters[idx:idx+dom_size]]
            idx += dom_size
        self.pgm.add_factor(Factor(self.pgm, rv, parents, table))

    def parse_string(self, text):
        text = HuginParser.rm_comments(text)
        result = None
        try:
            result = self.parser.parseString(text, parseAll=True)
        except ParseException as err:
            print(err)
        return result

    def parse(self):
        if self.fn is None:
            logger.warning('No filename given to parser')
            return None
        text = None
        logger.info("Start parsing ...")
        ts1 = time.clock()
        with open(self.fn, 'r') as ifile:
            text = ifile.read()
        self.parse_string(text)
        ts2 = time.clock()
        logger.info("Parsing took {:.3f} sec".format(ts2 - ts1))
        return self.pgm


def test(text):
    parser = HuginParser()
    return parser.parse_string(text)


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


def main(argv=None):
    description = 'Translate Bayesian net in Hugin .net/.hugin format format to ProbLog'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
    HuginParser.add_parser_arguments(parser)
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    parser = HuginParser(args)
    parser.run(args)


if __name__ == "__main__":
    sys.exit(main())

