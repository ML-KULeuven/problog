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
import math
from collections import Counter

class PGM(object):
    def __init__(self, cpds=None):
        """Probabilistic Graphical Model."""
        self.cpds = {}
        if cpds:
            for cpd in cpds:
                self.add(cpd)

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

    def compress_tables(self):
        """Analyze CPTs and join rows that have the same probability
        distribution."""
        cpds = [cpd.compress(self) for cpd in self.cpds.values()]
        return PGM(cpds)

    def to_hugin_net(self):
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

    def to_xdsl(self):
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

    def to_uai08(self):
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

    def to_problog(self, drop_zero=False, use_neglit=False, value_as_term=True):
        """Export PGM to ProbLog.
        """
        cpds = [cpd.to_CPT(self) for cpd in self.cpds_topological()]
        lines = ["%% ProbLog program",
                 "%% Created on {}\n".format(datetime.now())]
        lines += [cpd.to_ProbLog(self, drop_zero=drop_zero, use_neglit=use_neglit,
                                 value_as_term=value_as_term) for cpd in cpds]
        return '\n'.join(lines)

    def to_graphviz(self):
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


re_toundercore = re.compile(r"[\(\),\]\[ ]")
re_toremove = re.compile(r"""[^a-zA-Z0-9_]""")

boolean_values = [
  ['t','f'],
  ['true','false'],
  ['yes','no'],
  ['y','n'],
  ['pos','neg'],
  ['aye','nay']
]

class CPD(object):
    def __init__(self, rv, values, parents, detect_boolean=True):
        """Conditional Probability Distribution."""
        self.rv = rv
        self.values = values
        self.latent = False
        if parents is None:
            self.parents = []
        else:
            self.parents = parents
        self.booleantrue = None # Value that represents true
        if detect_boolean and len(self.values) == 2:
            for values in boolean_values:
                if values[0] == self.values[0].lower() and values[1] == self.values[1].lower():
                    self.booleantrue = 0
                    break
                elif values[1] == self.values[0].lower() and values[0] == self.values[1].lower():
                    self.booleantrue = 1
                    break

    def rv_clean(self, rv=None):
        if rv is None:
            rv = self.rv
        rv = re_toundercore.sub('_', rv)
        rv = re_toremove.sub('', rv)
        if not rv[0].islower():
            rv = rv[0].lower() + rv[1:]
            if not rv[0].islower():
                rv = "v"+rv
        return rv

    def has(self, rv):
        return rv == self.rv or rv in self.parents

    def to_CPT(self, pgm):
        return self

    def __str__(self):
        return '{} [{}]'.format(rv, ','.join(values))


class CPT(CPD):
    def __init__(self, rv, values, parents, table, *args, **kwargs):
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
        super(CPT, self).__init__(rv, values, parents, *args, **kwargs)
        if isinstance(table, list) or isinstance(table, tuple):
            self.table = {(): table}
        elif isinstance(table, dict):
            self.table = table
        else:
            raise ValueError('Unknown type (expected list, tuple or dict): {}'.format(type(table)))

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

    def compress(self, pgm):
        """Table to tree using the ID3 decision tree algorithm.

        From:
            {('f', 'f'): (0.2, 0.8),
             ('f', 't'): (0.2, 0.8),
             ('t', 'f'): (0.1, 0.9),
             ('t', 't'): (0.6, 0.4)}
        To:
            {('f', None): (0.2, 0.8),
             ('t', 'f'):  (0.1, 0.9),
             ('t', 't'):  (0.6, 0.4)}
        """
        # Traverse through the tree
        # First tuple is path with no value assignment and all rows in the table
        table = []
        for k,v in self.table.items():
            # Tuples allow hashing for IG
            table.append((k, tuple(v)))
        nodes = [(tuple([None]*len(self.parents)), table)]
        new_table = {}
        while len(nodes) > 0:
            curpath, node = nodes.pop()
            # All the same or all different? Then stop.
            k,v = zip(*node)
            c = Counter(v)
            if len(c.keys()) == 1:
                new_table[curpath] = node[0][1]
                continue
            if len(c.keys()) == len(v):
                for new_path, new_probs in node:
                    new_table[new_path] = new_probs
                continue
            # Find max information gain
            # ig_idx = self.maxinformationgainparent(node)
            ig_idx = 0
            ig_max = -1
            for parent_idx, parent in enumerate(self.parents):
                if curpath[parent_idx] is not None:
                    continue
                bins = {}
                for value in pgm.cpds[parent].values:
                    bins[value] = []
                for k,v in node:
                    bins[k[parent_idx]].append(v)
                ig = 0
                for bin_value, bin_labels in bins.items():
                    label_cnt = Counter(bin_labels)
                    h = -sum([cnt/len(bin_labels)*math.log(cnt/len(bin_labels),2) for cnt in label_cnt.values()])
                    ig += -len(bin_labels)/len(node)*h
                if ig > ig_max:
                    ig_max = ig
                    ig_idx = parent_idx
            # Create next nodes
            for value in pgm.cpds[self.parents[ig_idx]].values:
                newpath = [v for v in curpath]
                newpath[ig_idx] = value
                newnode = [tuple(newpath), []]
                for parent_values, prob in node:
                    if parent_values[ig_idx] == value:
                        newnode[1].append((parent_values, prob))
                nodes.append(newnode)
        return CPT(self.rv, self.values, self.parents, new_table)

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

    def to_ProbLogValue(self, value, value_as_term=True):
        if self.booleantrue is None:
            if value_as_term:
                return self.rv_clean()+'("'+str(value)+'")'
            else:
                return self.rv_clean()+'_'+self.rv_clean(str(value))
        else:
            if self.values[self.booleantrue] == value:
                return self.rv_clean()
            elif self.values[1-self.booleantrue] == value:
                return '\+'+self.rv_clean()
            else:
                raise Exception('Unknown value: {} = {}'.format(self.rv, value))

    def to_ProbLog(self, pgm, drop_zero=False, use_neglit=False, value_as_term=True):
        lines = []
        name = self.rv_clean()
        # if len(self.parents) > 0:
            # name += ' | '+' '.join([self.rv_clean(p) for p in self.parents])
        # table = sorted(self.table.items())
        table = self.table.items()
        # value_assignments = itertools.product(*[pgm.cpds[parent].value for parent in self.parents)

        for k, v in table:
            if self.booleantrue is not None and drop_zero and v[self.booleantrue] == 0.0 and not use_neglit:
                continue
            head_lits = []
            if self.booleantrue is None:
                for idx,vv in enumerate(v):
                    if not (drop_zero and vv == 0.0):
                        head_lits.append('{}::{}'.format(vv, self.to_ProbLogValue(self.values[idx], value_as_term)))
            else:
                if drop_zero and v[self.booleantrue] == 0.0 and use_neglit:
                    head_lits.append(self.to_ProbLogValue(self.values[1-self.booleantrue], value_as_term))
                elif v[self.booleantrue] == 1.0:
                    head_lits.append(self.to_ProbLogValue(self.values[self.booleantrue], value_as_term))
                else:
                    head_lits.append('{}::{}'.format(v[self.booleantrue], self.to_ProbLogValue(self.values[self.booleantrue],value_as_term)))
            body_lits = []
            for parent,parent_value in zip(self.parents, k):
                if parent_value is not None:
                    parent_cpd = pgm.cpds[parent]
                    body_lits.append(parent_cpd.to_ProbLogValue(parent_value, value_as_term))
            if len(body_lits) > 0:
                lines.append('{} :- {}.'.format('; '.join(head_lits), ', '.join(body_lits)))
            else:
                lines.append('{}.'.format('; '.join(head_lits)))
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
