#!/usr/bin/env python3
# encoding: utf-8
"""
xmlbif2problog.py

http://www.cs.cmu.edu/~fgcozman/Research/InterchangeFormat/

Created by Wannes Meert.
Copyright (c) 2017 KU Leuven. All rights reserved.
"""
from __future__ import print_function

import os
import sys
import argparse
import itertools
import logging
import xml.etree.ElementTree as ET

from bn2problog import BNParser

try:
    from problog.pgm.cpd import Variable, Factor, PGM
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from problog.pgm.cpd import Variable, Factor, PGM


logger = logging.getLogger('be.kuleuven.cs.dtai.problog.bn2problog')


class XMLBIFParser(BNParser):
    def __init__(self, args):
        super(XMLBIFParser, self).__init__(args)
        self.domains = {}
        self.potentials = []

    def parse(self):
        if self.fn is None:
            return None
        with open(self.fn) as ifile:
            tree = ET.parse(ifile)
        root = tree.getroot()

        self.check_version(root)
        networks = root.findall('NETWORK')
        if len(networks) == 0:
            logger.error("No <NETWORK> tag found.", halt=True)
        if len(networks) > 1:
            logger.warning("Multiple networks found, only the first one is used")
        network = networks[0]
        self.parse_name(network)
        self.parse_properties(network)
        self.parse_domains(network)
        for cpt in network.findall('DEFINITION'):
            self.parse_cpt(cpt)
        return self.pgm

    def check_version(self, root):
        version = float(root.get('VERSION'))
        if version < 0.3:
            logger.error("Outdated version ({}), expects at least 0.3".format(version), halt=True)

    def parse_name(self, root):
        try:
            name = root.find('NAME').text
            self.pgm.name = name
        except Exception:
            pass

    def parse_properties(self, root):
        try:
            props = root.findall('PROPERTY')
            for prop in props:
                self.pgm.comments.append(prop.text)
        except Exception:
            pass

    def parse_domains(self, root):
        for cpt in root.findall('VARIABLE'):
            rv = cpt.find('NAME').text
            node_type = cpt.get("TYPE")
            if node_type != "nature":
                logger.error("Only probabilistic variables are supported. Found type {} for variable {}".format(node_type, name), halt=True)
            states = cpt.findall('OUTCOME')
            values = [state.text for state in states]
            self.domains[rv] = values
            self.pgm.add_var(Variable(rv, values, detect_boolean=self.detect_bool, force_boolean=self.force_bool))

    def parse_cpt(self, cpt):
        rv = cpt.find('FOR').text
        if rv not in self.domains:
            logger.error('Domain for {} not defined.'.format(rv), halt=True)
            sys.exit(1)
        values = self.domains[rv]
        parents = cpt.findall('GIVEN')
        if parents is None:
            parents = []
        else:
            parents = [parent.text for parent in parents]
        parameters = [float(p) for p in cpt.find('TABLE').text.strip().split()]
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
    parser = argparse.ArgumentParser(description='Translate Bayesian net in XMLBIF format format to ProbLog')
    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')
    parser.add_argument('--quiet', '-q', action='count', default=0, help='Quiet output')
    XMLBIFParser.add_parser_arguments(parser)
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.INFO - 10 * (args.verbose - args.quiet), logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    parser = XMLBIFParser(args)
    parser.run(args)


if __name__ == "__main__":
    sys.exit(main())
