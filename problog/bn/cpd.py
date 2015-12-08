"""
ProbLog command-line interface.

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

class PGM(object):
    def __init__(self):
        self.cpds = {}

    def add(self, cpd):
        if cpd.rv in self.cpds:
            self.cpds[cpd.rv] += cpd
        else:
            self.cpds[cpd.rv] = cpd

    def __str__(self):
        cpds = [cpd.to_CPT(self) for cpd in self.cpds.values()]
        return '\n'.join([str(cpd) for cpd in cpds])


class CPD(object):
    def __init__(self, rv, values, parents):
        self.rv = rv
        self.values = values
        if parents is None:
            self.parents = []
        else:
            self.parents = parents

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

    def __str__(self):
        lines = []
        table = sorted(self.table.items())
        for k, v in table:
            lines.append('{}: {}'.format(k, v))
        table = '\n'.join(lines)
        parents = ''
        if len(self.parents) > 0:
            parents = ' -- {}'.format(','.join(self.parents))
        return 'CPT {} [{}]{}\n{}'.format(self.rv, ','.join([str(v) for v in self.values]), parents, table)


class OrCPT(CPD):
    def __init__(self, rv, parentvalues=None):
        super(OrCPT, self).__init__(rv, [False, True], [])
        if parentvalues is None:
            self.parentvalues = []
        else:
            self.parentvalues = parentvalues

    def add(self, parentvalues):
        """Add list of tupes [('a', 1)]."""
        self.parentvalues += parentvalues

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
