"""
problog.maxsat - Interface to MaxSAT solvers
--------------------------------------------

Interface to MaxSAT solvers.

..
    Part of the ProbLog distribution.

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

from .util import mktempfile, subprocess_check_output
from . import root_path
from .errors import ProbLogError


class UnsatisfiableError(ProbLogError):

    def __init__(self):
        ProbLogError.__init__(self, 'No solution exists that satisfies the constraints.')



class MaxSATSolver(object):

    def __init__(self, command):
        self.command = command

    @property
    def extension(self):
        return 'cnf'

    def prepare_input(self, formula, **kwargs):
        return formula.to_dimacs(weighted=int, **kwargs)

    def process_output(self, output):
        for line in output.split('\n'):
            if line.startswith('v '):
                return list(map(int, line.split()[1:-1]))
        raise UnsatisfiableError()

    def call_process(self, inputf):
        filename = mktempfile('.' + self.extension)
        with open(filename, 'w') as f:
            f.write(inputf)
        return subprocess_check_output(self.command + [filename])

    def evaluate(self, formula, **kwargs):
        inputf = self.prepare_input(formula, **kwargs)
        output = self.call_process(inputf)
        result = self.process_output(output)
        return result


class MIPMaxSATSolver(MaxSATSolver):

    def __init__(self, command):
        MaxSATSolver.__init__(self, command)

    @property
    def extension(self):
        return 'lp'

    def prepare_input(self, formula, **kwargs):
        return formula.to_lp(**kwargs)


class SCIPSolver(MIPMaxSATSolver):

    def __init__(self):
        MaxSATSolver.__init__(self, ['scip', '-f'])

    def process_output(self, output):
        facts = set()
        in_the_zone = False
        for line in output.split('\n'):
            line = line.strip()
            if line.startswith('objective value'):
                in_the_zone = True
            elif in_the_zone:
                if not line:
                    return list(facts)
                else:
                    facts.add(int(line.split()[0][1:]))
        raise UnsatisfiableError()


def get_solver(prefer=None):
    if prefer == 'scip':
        return SCIPSolver()
    elif prefer == 'sat4j':
        return MaxSATSolver(['java', '-jar',
                             root_path('problog', 'bin', 'java', 'sat4j-maxsat.jar')])
    else:
        return MaxSATSolver(['maxsatz'])


def get_available_solvers():
    # TODO check whether they are actually available
    return ['maxsatz', 'scip', 'sat4j']
