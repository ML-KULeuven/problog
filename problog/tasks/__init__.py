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


problog_tasks = {}
problog_tasks['prob'] = 'problog.tasks.probability'
problog_tasks['mpe'] = 'problog.tasks.mpe'
problog_tasks['sample'] = 'problog.tasks.sample'
problog_tasks['ground'] = 'problog.tasks.ground'
problog_tasks['lfi'] = '../../learning/lfi.py'

problog_default_task = 'prob'

import os
import imp


def run_task(argv):
    if argv[0] in problog_tasks:
        task = argv[0]
        args = argv[1:]
    else:
        task = problog_default_task
        args = argv
    return load_task(task).main(args)


def load_task(name):
    return _load_extension(problog_tasks[name])


def _load_extension(filename):
    if filename.endswith('.py'):
        filename = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
        (path, name) = os.path.split(filename)
        (name, ext) = os.path.splitext(name)
        (extfile, filename, data) = imp.find_module(name, [path])
        return imp.load_module(name, extfile, filename, data)
    else:
        mod = __import__(filename)
        components = filename.split('.')
        for c in components[1:]:
            mod = getattr(mod, c)
        return mod
