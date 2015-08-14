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
problog_tasks['prob'] = 'probability.py'
problog_tasks['mpe'] = 'mpe/mpe.py'
problog_tasks['lfi'] = '../learning/lfi.py'
problog_tasks['sample'] = 'sample/sample.py'
problog_tasks['ground'] = 'ground.py'

problog_default_task = 'prob'

import os
import imp

def load_extension(filename):
    filename = os.path.abspath(os.path.join(os.path.dirname(__file__), filename))
    (path, name) = os.path.split(filename)
    (name, ext) = os.path.splitext(name)

    with open(filename, 'r') as extfile:
        (file, filename, data) = imp.find_module(name, [path])
        return imp.load_module(name, file, filename, data)
