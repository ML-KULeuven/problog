#! /usr/bin/env python
"""
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

import os
import sys
import json

sys.setrecursionlimit(10000)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from problog.errors import ProbLogError


def print_result_prob(d, output, precision=8):
    """Pretty print result.

    :param d: result from run_problog
    :param output: output file
    :param precision:
    :return:
    """
    result = {}
    success, d = d
    if success:
        result['SUCCESS'] = True
        result['probs'] = [[str(n), round(p, precision), n.loc[1], n.loc[2]] for n, p in d.items()]
    else:
        result['SUCCESS'] = False
        result['err'] = process_error(d)
    print (200, 'application/json', json.dumps(result), file=output)
    return 0


def print_result_mpe(d, output, precision=8):
    """Pretty print result.

    :param d: result from run_problog
    :param output: output file
    :param precision:
    :return:
    """
    result = {}
    success, d = d
    if success:
        result['SUCCESS'] = True
        result['atoms'] = list(map(lambda n: (str(-n), False) if n.is_negated() else (str(n), True), d))
    else:
        result['SUCCESS'] = False
        result['err'] = process_error(d)
    print (200, 'application/json', json.dumps(result), file=output)
    return 0


def print_result_sample(d, output, **kwdargs):
    """Pretty print result.

    :param d: result from run_problog
    :param output: output file
    :param precision:
    :return:
    """
    result = {}
    success, d = d
    if success:
        result['SUCCESS'] = True
        result['results'] = [[(str(k), str(v)) for k, v in dc.items()] for dc in d]
    else:
        result['SUCCESS'] = False
        result['err'] = process_error(d)
    print (200, 'application/json', json.dumps(result), file=output)
    return 0


def process_error(err):
    if isinstance(err, ProbLogError):
        return vars(err)
    else:
        return {'message': 'An unexpected error has occurred (%s).' % err}


def main(args):
    task = args[0]
    args = list(args[:-1]) + ['-o', args[-1]]
    from problog.tasks import run_task

    if task == 'mpe':
        print_result = print_result_mpe
    elif task == 'sample':
        print_result = print_result_sample
    else:
        print_result = print_result_prob

    run_task(args, print_result)


if __name__ == '__main__':
    main(sys.argv[1:])
