"""
Conditional Probability Distributions (CPD) to expert ProbLog to a
Probabilistic Graphical Model (PGM).

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

import itertools
from datetime import datetime
import re
from functools import reduce
import sys

class PGM(object):
    def __init__(self):
        """Probabilistic Graphical Model."""
        self.cpds = {}

    def add(self, cpd):
        """Add a CPD."""
        if cpd.rv in self.cpds:
            self.cpds[cpd.rv] += cpd
        else:
            self.cpds[cpd.rv] = cpd

    def cpds_topological(self):
        """Return the CPDs in a topological order."""
        # Links from parent-node to child-node
        links = dict()
        for cpd in self.cpds.values():
            for parent in cpd.parents:
                if parent in links:
                    links[parent].add(cpd.rv)
                else:
                    links[parent] = set([cpd.rv])
        # print(links)
        all = set(links.keys())
        nonroot = reduce(set.union, links.values())
        root = all - nonroot
        # print(root)
        queue = root
        visited = set()
        cpds = []
        while len(queue) > 0:
            for rv in queue:
                if rv in visited:
                    continue
                all_parents_visited = True
                try:
                    cpd = self.cpds[rv]
                except KeyError as exc:
                    print('Error: random variable has no CPD associated:')
                    print(exc)
                    # print('\n'.join(self.cpds.keys()))
                    sys.exit(1)
                for parent_rv in cpd.parents:
                    if parent_rv not in visited:
                        all_parents_visited = False
                        break
                if all_parents_visited:
                    cpds.append(cpd)
                    visited.add(rv)
            queue2 = set()
            for rv in queue:
                if rv in links:
                    for next_rv in links[rv]:
                        queue2.add(next_rv)
            queue = queue2
        return cpds

    def marginalizeLatentVariables(self):
        marg_sets = []
        # Find nodes with only latent variables that are only parent for that node
        for cpd in self.cpds.values():
            # print('cpd: {} | {}'.format(cpd.rv, cpd.parents))
            if len(cpd.parents) == 0:
                continue
            all_latent = True
            for parent in cpd.parents:
                if not self.cpds[parent].latent:
                    all_latent = False
            if not all_latent:
                continue
            marg_sets.append((cpd.rv, cpd.parents))
        print(marg_sets)
        # Eliminate
        for (child_rv, parent_rvs) in marg_sets:
            cpds = [self.cpds[child_rv]] + [self.cpds[rv] for rv in parent_rvs]
            cpd = CPT.marginalize(cpds, parent_rvs)
            if not cpd is None:
                print(cpd)
                self.cpds[child_rv] = cpd
                for rv in parent_rvs:
                    del self.cpds[rv]

    def hugin_net(self):
        """Export PGM to the Hugin net format.
        http://www.hugin.com/technology/documentation/api-manuals
        """
        cpds = [cpd.to_CPT(self) for cpd in self.cpds_topological()]
        lines = ["%% Hugin Net format",
                 "%% Created on {}\n".format(datetime.now()),
                 "%% Network\n",
                 "net {",
                 "  node_size = (50,50);",
                 "}\n",
                 "%% Nodes\n"]
        lines += [cpd.to_HuginNetNode() for cpd in cpds]
        lines += ["%% Potentials\n"]
        lines += [cpd.to_HuginNetPotential() for cpd in cpds]
        return '\n'.join(lines)

    def xdsl(self):
        """Export PGM to the XDSL format defined by SMILE.
        https://dslpitt.org/genie/wiki/Appendices:_XDSL_File_Format_-_XML_Schema_Definitions
        """
        cpds = [cpd.to_CPT(self) for cpd in self.cpds_topological()]
        lines = ['<?xml version="1.0" encoding="ISO-8859-1" ?>',
                 '<smile version="1.0" id="Aa" numsamples="1000">',
                 '  <nodes>']
        lines += [cpd.to_XdslCpt() for cpd in cpds]
        lines += ['  </nodes>',
                  '  <extensions>',
                  '    <genie version="1.0" app="ProbLog" name="Network1" faultnameformat="nodestate">']
        lines += [cpd.to_XdslNode() for cpd in cpds]
        lines += ['    </genie>',
                  '  </extensions>',
                  '</smile>']
        return '\n'.join(lines)

    def uai08(self):
        """Export PGM to the format used in the UAI 2008 competition.
        http://graphmod.ics.uci.edu/uai08/FileFormat
        """
        cpds = [cpd.to_CPT(self) for cpd in self.cpds_topological()]
        number_variables = str(len(cpds))
        domain_sizes = [str(len(cpd.values)) for cpd in cpds]
        number_functions = str(len(cpds))
        lines = ['BAYES',
                 number_variables,
                 ' '.join(domain_sizes),
                 number_functions]
        lines += [cpd.to_Uai08Preamble(cpds) for cpd in cpds]
        lines += ['']
        lines += [cpd.to_Uai08Function() for cpd in cpds]
        return '\n'.join(lines)

    def graphviz(self):
        """Export PGM to Graphviz dot format.
        http://www.graphviz.org
        """
        lines = ['digraph bayesnet {']
        for cpd in self.cpds.values():
            lines.append('  {} [label="{}"];'.format(cpd.rv_clean(), cpd.rv))
            for p in cpd.parents:
                lines.append('  {} -> {}'.format(cpd.rv_clean(p), cpd.rv_clean()))
        lines += ['}']
        return '\n'.join(lines)

    def __str__(self):
        cpds = [cpd.to_CPT(self) for cpd in self.cpds_topological()]
        return '\n'.join([str(cpd) for cpd in cpds])


re_clean = re.compile(r"[\(\),\]\[ ]")

class CPD(object):
    def __init__(self, rv, values, parents):
        """Conditional Probability Distribution."""
        self.rv = rv
        self.values = values
        self.latent = False
        if parents is None:
            self.parents = []
        else:
            self.parents = parents

    def rv_clean(self, rv=None):
        if rv is None:
            rv = self.rv
        return re_clean.sub('_', rv)

    def has(self, rv):
        return rv == self.rv or rv in self.parents

    def to_CPT(self, pgm):
        return self

    def __str__(self):
        return '{} [{}]'.format(rv, ','.join(values))


class CPT(CPD):
    def __init__(self, rv, values, parents, table):
        """Conditional Probability Table with discrete probabilities.

        :param rv: random variable
        :param values: random variable domain
        :param parents: parent random variables
        :param table:

        a = CPT('a', ['f','t'], [], [0.4,0.6])
        b = CPT('b', ['f','t'], [a],
                {('f',): [0.2,0.8],
                 ('t',): [0.7,0.3]})
        """
        super(CPT, self).__init__(rv, values, parents)
        if isinstance(table, list) or isinstance(table, tuple):
            self.table = {(): table}
        else:
            self.table = table

    @staticmethod
    def marginalize(cpds, margvars):
        children = set([cpd.rv for cpd in cpds])
        print('children: {}'.format(children))
        child = children - margvars
        print('child: {}'.format(child))
        parents = reduce(set.union, [cpd.parents for cpd in cpds])
        parents -= margvars
        print('parents: {}'.format(parents))

        return None

    def to_HuginNetNode(self):
        lines = ["node {} {{".format(self.rv_clean()),
                 "  label = \"{}\";".format(self.rv),
                 "  position = (100,100);",
                 "  states = ({});".format(' '.join(['"{}"'.format(v) for v in self.values])),
                 "}\n"]
        return '\n'.join(lines)

    def to_HuginNetPotential(self):
        name = self.rv_clean()
        if len(self.parents) > 0:
            name += ' | '+' '.join([self.rv_clean(p) for p in self.parents])
        lines = ['potential ({}) {{'.format(name),
                 '  % '+' '.join([str(v) for v in self.values]),
                 '  data = (']
        table = sorted(self.table.items())
        for k, v in table:
            lines.append('    '+' '.join([str(vv) for vv in v])+' % '+' '.join([str(kk) for kk in k]))
        lines += ['  );',
                  '}\n']
        return '\n'.join(lines)

    def to_XdslCpt(self):
        lines = ['    <cpt id="{}">'.format(self.rv_clean())]
        for v in self.values:
            lines.append('      <state id="{}" />'.format(v))
        if len(self.parents) > 0:
            lines.append('      <parents>{}</parents>'.format(' '.join([self.rv_clean(p) for p in self.parents])))
        table = sorted(self.table.items())
        probs = ' '.join([str(value) for k,values in table for value in values])
        lines.append('      <probabilities>{}</probabilities>'.format(probs))
        lines.append('    </cpt>')
        return '\n'.join(lines)

    def to_XdslNode(self):
        lines = [
            '      <node id="{}">'.format(self.rv_clean()),
            '        <name>{}</name>'.format(self.rv),
            '        <interior color="e5f6f7" />',
			'        <outline color="000080" />',
            '        <font color="000000" name="Arial" size="8" />',
            '        <position>100 100 150 150</position>',
            '      </node>']
        return '\n'.join(lines)

    def to_Uai08Preamble(self, cpds):
        function_size = 1 + len(self.parents)
        rvToIdx = {}
        for idx, cpd in enumerate(cpds):
            rvToIdx[cpd.rv] = idx
        variables = [str(rvToIdx[rv]) for rv in self.parents] + [str(rvToIdx[self.rv])]
        return '{} {}'.format(function_size, ' '.join(variables))

    def to_Uai08Function(self):
        number_entries = str(len(self.table)*len(self.values))
        lines = [number_entries]
        table = sorted(self.table.items())
        for k, v in table:
            lines.append(' '+' '.join([str(vv) for vv in v]))
        lines.append('')
        return '\n'.join(lines)

    def __str__(self):
        lines = []
        table = sorted(self.table.items())
        for k, v in table:
            lines.append('{}: {}'.format(k, v))
        table = '\n'.join(lines)
        parents = ''
        if len(self.parents) > 0:
            parents = ' | {}'.format(','.join(self.parents))
        return 'CPT ({}{}) = {}\n{}'.format(self.rv, parents, ','.join([str(v) for v in self.values]), table)


class OrCPT(CPD):
    def __init__(self, rv, parentvalues=None):
        super(OrCPT, self).__init__(rv, [False, True], set())
        if parentvalues is None:
            self.parentvalues = []
        else:
            self.parentvalues = parentvalues
        self.parents.update([pv[0] for pv in self.parentvalues])

    def add(self, parentvalues):
        """Add list of tupes [('a', 1)]."""
        self.parentvalues += parentvalues
        self.parents.update([pv[0] for pv in parentvalues])

    def to_CPT(self, pgm):
        parents = sorted(list(set([pv[0] for pv in self.parentvalues])))
        table = dict()
        parent_values = [pgm.cpds[parent].values for parent in parents]
        for keys in itertools.product(*parent_values):
            is_true = False
            for parent, value in zip(parents, keys):
                if (parent, value) in self.parentvalues:
                    is_true = True
            if is_true:
                table[keys] = [0.0, 1.0]
            else:
                table[keys] = [1.0, 0.0]
        return CPT(self.rv, self.values, parents, table)

    def __add__(self, other):
        return OrCPT(self.rv, self.parentvalues + other.parentvalues)

    def __str__(self):
        table = '\n'.join(['{}'.format(pv) for pv in self.parentvalues])
        parents = ''
        if len(self.parents) > 0:
            parents = ' -- {}'.format(','.join(self.parents))
        return 'OrCPT {} [{}]{}\n{}'.format(self.rv, ','.join(self.values), parents, table)
