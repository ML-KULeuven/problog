from __future__ import print_function

import argparse
import time

from ..program import SimpleProgram, PrologFile
from ..engine import DefaultEngine
from ..formula import LogicFormula, LogicDAG
from .. import get_evaluatables, get_evaluatable


def argparser():
    ap = argparse.ArgumentParser()

    # TODO support more options of task 'prob' (e.g. -k)

    ap.add_argument('filename', nargs='+', help='Name of file to run (can be multiple).')
    ap.add_argument('-b', '--base', action='append', default=[], help='Additional models to load.')
    ap.add_argument('-n', '--repeat', type=int, default=1, help='Repeat each run this many times.')
    ap.add_argument('-k', choices=get_evaluatables())

    return ap


def main(argv):
    parser = argparser()
    args = parser.parse_args(argv)

    first = True
    for filename in args.filename:
        for run in range(0, args.repeat):
            timers = process(filename, base=args.base, ktype=args.k)
            if first:
                print('filename', timers.header, 'total', sep=';')
                first = False
            print(filename, timers, timers.total, sep=';')


class Timer(object):

    def __init__(self, name):
        self.name = name
        self._start_time = None
        self._end_time = None
        self._err_name = None

    def __enter__(self):
        self._start_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self._end_time = None
            self._err_name = exc_type.__name__
        else:
            self._end_time = time.time()
        return

    @property
    def elapsed_time(self):
        if self._err_name is not None:
            return self._err_name
        else:
            return self._end_time - self._start_time

    @property
    def elapsed_time_string(self):
        if self._err_name is not None:
            return self._err_name
        else:
            return '%.6f' % (self._end_time - self._start_time)


class TimerCollection(object):

    def __init__(self):
        self.timers = []

    def new(self, name):
        tmr = Timer(name)
        self.timers.append(tmr)
        return tmr

    @property
    def header(self):
        return ';'.join([tmr.name for tmr in self.timers])

    def __str__(self):
        return ';'.join([tmr.elapsed_time_string for tmr in self.timers])

    @property
    def total(self):
        try:
            return sum(tmr.elapsed_time for tmr in self.timers)
        except TypeError:
            return 'ERROR'

    def __len__(self):
        return len(self.timers)


def process(filename, base=None, ktype=None):
    if base is None:
        base = []

    timers = TimerCollection()

    # Step 1: parse the model
    with timers.new('parse'):
        model = SimpleProgram()
        for i, fn in enumerate([filename] + base):
            filemodel = PrologFile(fn)
            for line in filemodel:
                model += line
            if i == 0:
                model.source_root = filemodel.source_root

    # Step 2: compile the model into a database
    with timers.new('load'):
        engine = DefaultEngine()
        database = engine.prepare(model)

    # Step 3: ground all
    with timers.new('ground'):
        ground_program = LogicFormula.create_from(database)

    # Step 4: break cycles
    with timers.new('cycles'):
        if ktype == 'fsdd':
            ground_formula = ground_program
        else:
            ground_formula = LogicDAG.create_from(ground_program)

    # Step 5: compile
    with timers.new('compile'):
        knowledge = get_evaluatable(ktype).create_from(ground_formula)

    # Step 6: evaluate
    with timers.new('evaluate'):
        results = knowledge.evaluate()

    return timers
