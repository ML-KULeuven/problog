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

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from problog.util import load_module
from problog import version

problog_tasks = {}
problog_tasks['prob'] = 'problog.tasks.probability'
problog_tasks['mpe'] = 'problog.tasks.mpe'
problog_tasks['sample'] = 'problog.tasks.sample'
problog_tasks['ground'] = 'problog.tasks.ground'
problog_tasks['lfi'] = 'problog.learning.lfi'
problog_tasks['explain'] = 'problog.tasks.explain'
problog_tasks['web'] = 'problog.web.server'
problog_tasks['dt'] = 'problog.tasks.dtproblog'
problog_tasks['shell'] = 'problog.tasks.shell'
problog_tasks['parse'] = 'problog.parser'
problog_tasks['map'] = 'problog.tasks.map'

problog_default_task = 'prob'


def run_task(argv):
    """Execute a task in ProbLog.
    If the first argument is a known task name, that task is executed.
    Otherwise the default task is executed.

    :param argv: list of arguments for the task
    :return: result of the task (typically None)
    """
    if len(argv) > 0 and argv[0] in problog_tasks:
        task = argv[0]
        args = argv[1:]
    else:
        task = problog_default_task
        args = argv
    return load_task(task).main(args)


def load_task(name):
    """Load the module for executing the given task.

    :param name: task name
    :type name: str
    :return: loaded module
    :rtype: module
    """
    return load_module(problog_tasks[name])


def main(argv=None):
    """Main function.
    :param argv: command line arguments
    """
    if argv is None:
        argv = sys.argv[1:]

    if len(argv) > 0:
        if argv[0] == 'install':
            from .. import setup
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
        elif argv[0] == '--version':
            print (version.version)
            return
        else:
            return run_task(argv)
    else:
        return run_task(argv)


if __name__ == '__main__':
    main(sys.argv[1:])
