#! /usr/bin/env python
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

import os
import sys


def main(argv):
    """Main function.
    :param argv: command line arguments
    """

    from tasks import problog_tasks, problog_default_task, load_extension

    if len(argv) > 0:
        if argv[0] == 'install':
            from problog import setup
            setup.install()
            return
        elif argv[0] == 'info':
            from problog.core import list_transformations
            list_transformations()
            return
        elif argv[0] == 'unittest':
            import unittest
            test_results = unittest.TextTestResult(sys.stderr, False, 1)
            unittest.TestLoader().discover(os.path.dirname(__file__)).run(test_results)
            return
        elif argv[0] in problog_tasks:
            extension = load_extension(problog_tasks[argv[0]])
            return extension.main(argv[1:])
        else:
            extension = load_extension(problog_tasks[problog_default_task])
            return extension.main(argv)

if __name__ == '__main__':
    main(sys.argv[1:])
