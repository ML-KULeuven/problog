#!/usr/bin/env python3
# encoding: utf-8
"""
hugin2problog.py

Created by Wannes Meert on 23-02-2016.
Copyright (c) 2016 KU Leuven. All rights reserved.
"""
from __future__ import print_function

import os
import sys
import argparse
import itertools
import logging
import xml.etree.ElementTree as ET

from bn2problog import BNParser

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from problog.pgm.cpd import Variable, Factor, PGM



logger = logging.getLogger('be.kuleuven.cs.dtai.problog.bn2problog')


class SmileParser(BNParser):
    def __init__(self, args):
        super(SmileParser, self).__init__(args)
        self.domains = {}
        self.potentials = []

    def parse(self):
        if self.fn is None:
            return None
        with open(self.fn) as ifile:
            tree = ET.parse(ifile)
        root = tree.getroot()

        self.parse_domains(root)
        for cpt in root.find("nodes").findall('cpt'):
            self.parse_cpt(cpt)
        return self.pgm

    def parse_domains(self, root):
        for cpt in root.find("nodes").findall('cpt'):
            rv = cpt.get('id')
            states = cpt.findall('state')
            values = [state.get('id') for state in states]
            self.domains[rv] = values
            self.pgm.add_var(Variable(rv, values, detect_boolean=self.detect_bool, force_boolean=self.force_bool))

    def parse_cpt(self, cpt):
        rv = cpt.get('id')
        if rv not in self.domains:
            logger.error('Domain for {} not defined.'.format(rv), halt=True)
            sys.exit(1)
        values = self.domains[rv]
        parents = cpt.find('parents')
        if parents is None:
            parents = []
        else:
            parents = parents.text.split()
        parameters = [float(p) for p in cpt.find('probabilities').text.split()]
        if len(parents) == 0:
            table = parameters
            self.pgm.add_factor(Factor(self.pgm, rv, parents, table))
            return
        parent_domains = []
        for parent in parents:
            parent_domains.append(self.domains[parent])
        dom_size = len(values)
        table = {}
        idx = 0
        for val_assignment in itertools.product(*parent_domains):
            table[val_assignment] = parameters[idx:idx + dom_size]
            idx += dom_size
        self.pgm.add_factor(Factor(self.pgm, rv, parents, table))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Translate Bayesian net in Smile/Genie .xdsl format format to ProbLog')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
    SmileParser.add_parser_arguments(parser)
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    parser = SmileParser(args)
    parser.run(args)


if __name__ == "__main__":
    sys.exit(main())
