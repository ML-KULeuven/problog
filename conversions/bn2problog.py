#!/usr/bin/env python3
# encoding: utf-8
"""
bn2problog.py

Created by Wannes Meert on 06-03-2017.
Copyright (c) 2016 KU Leuven. All rights reserved.
"""
from __future__ import print_function

import sys
import os
import argparse
import itertools
import logging
import abc

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from problog.pgm.cpd import PGM

logger = logging.getLogger('be.kuleuven.cs.dtai.problog.bn2problog')


class BNParser:
    def __init__(self, args):
        """BNParser abstract base class."""
        self.fn = args.input
        self.pgm = PGM()

        if args.force_bool:
            self.force_bool = args.force_bool
        if args.nobooldetection:
            self.detect_bool = False
        if args.drop_zero:
            self.drop_zero = args.drop_zero
        if args.use_neglit:
            self.use_neglit = args.use_neglit

        self.force_bool = False
        self.detect_bool = True
        self.drop_zero = False
        self.use_neglit = False

    @abc.abstractmethod
    def parse(self):
        pass

    @staticmethod
    def add_parser_arguments(parser):
        parser.add_argument('--forcebool', dest='force_bool', action='store_true',
                            help='Force binary nodes to be represented as Boolean predicates (0=f, 1=t)')
        parser.add_argument('--nobooldetection', action='store_true',
                            help='Do not try to detect Boolean predicates (true/false, yes/no, ...)')
        parser.add_argument('--dropzero', dest='drop_zero', action='store_true',
                            help='Drop zero probabilities (if possible)')
        parser.add_argument('--useneglit', dest='use_neglit', action='store_true',
                            help='Use negative head literals')
        parser.add_argument('--allowdisjunct', action='store_true',
                            help='Allow disjunctions in the body')
        parser.add_argument('--valueinatomname', action='store_false',
                            help='Add value to atom name instead as a term (this removes invalid characters, '
                                 'be careful that clean values do not overlap)')
        parser.add_argument('--adisfunction', action='store_true',
                            help='Consider all ADs to represent functions of mutual exclusive conditions (like '
                                 'in a Bayesian net)')
        parser.add_argument('--compress', action='store_true',
                            help='Compress tables')
        parser.add_argument('--split',
                            help='Comma-separated list of variable names to split on')
        parser.add_argument('--split-output', dest='splitoutput', action='store_true',
                            help='Create one output file per connected network')
        parser.add_argument('--output', '-o',
                            help='Output file')
        parser.add_argument('--output-format', default='problog',
                            help='Output format (\'problog\', \'uai\', \'hugin\', \'xdsl\', \'xmlbif\')')
        parser.add_argument('input',
                            help='Input file')

    def run(self, args):
        pgm = self.parse()
        if args.compress:
            pgm = pgm.compress_tables(allow_disjunct=args.allowdisjunct)
        if args.split:
            pgm = pgm.split_topological(set(args.split.split(',')))
        if pgm is None:
            logger.error('Could not build PGM structure')
            sys.exit(1)
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
                elif args.output_format in ["hugin", "net"]:
                    print(pgm.to_hugin_net(), file=ofile)
                elif args.output_format in ["smile", "xdsl"]:
                    print(pgm.to_xdsl(), file=ofile)
                elif args.output_format in ["xml", "xmlbif"]:
                    print(pgm.to_xmlbif(), file=ofile)
                elif args.output_format in ["graphiz", "dot"]:
                    print(pgm.to_graphviz(), file=ofile)
                else:
                    print(pgm.to_problog(drop_zero=self.drop_zero, use_neglit=self.use_neglit,
                                         value_as_term=args.valueinatomname,
                                         ad_is_function=args.adisfunction), file=ofile)
            finally:
                if args.output:
                    ofile.close()


def main(argv=None):
    description = 'Translate Bayesian net to ProbLog'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
    parser.add_argument('--input-format', help='Input type (\'smile\', \'hugin\', \'uai\', \'xmlbif\')')
    BNParser.add_parser_arguments(parser)
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    parser = None
    input_format = None
    if args.input_format:
        input_format = args.input_format
    else:
        # try to infer
        _, ext = os.path.splitext(args.input)
        if ext in ['.uai']:
            input_format = 'uai'
        elif ext in ['.net']:
            input_format = 'hugin'
        elif ext in ['.xdsl']:
            input_format = 'smile'
        elif ext in ['.xml']:
            input_format = 'xmlbif'

    if input_format is None:
        logger.error('No file format given or detected (.uai, .net, .xdsl, .xml).')
        sys.exit(1)

    if input_format in ['uai', 'uai08']:
        from uai2problog import UAIParser
        parser = UAIParser(args)
    elif input_format in ['hugin', 'net']:
        from hugin2problog import HuginParser
        parser = HuginParser(args)
    elif input_format in ['smile', 'xdsl']:
        from smile2problog import SmileParser
        parser = SmileParser(args)
    elif input_format in ['xml', 'xmlbif']:
        from xmlbif2problog import XMLBIFParser
        parser = XMLBIFParser(args)
    else:
        logger.error("Unknown input format: {}".format(input_format))
        sys.exit(1)

    if parser:
        parser.run(args)


if __name__ == "__main__":
    sys.exit(main())
