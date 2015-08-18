"""
Module name
"""

from __future__ import print_function

from .util import mktempfile
from . import root_path


import subprocess


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
        return None

    def call_process(self, inputf):
        filename = mktempfile('.' + self.extension)
        with open(filename, 'w') as f:
            f.write(inputf)
        return subprocess.check_output(self.command + [filename])

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


def get_solver(prefer=None):
    if prefer == 'scip':
        return SCIPSolver()
    elif prefer == 'sat4j':
        return MaxSATSolver(['java', '-jar',
                             root_path('problog', 'bin', 'java', 'sat4j-maxsat.jar')])
    else:
        return SCIPSolver()


def get_available_solvers():
    # TODO check whether they are available
    return ['scip', 'sat4j']