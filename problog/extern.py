"""
problog.extern - Calling Python from ProbLog
--------------------------------------------

Interface for calling Python from ProbLog.

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
from .engine_builtin import problog_export, problog_export_nondet, problog_export_raw, problog_export_builtin

from problog.logic import Constant


def problog_export_class(cls):
    prefix = cls.__name__.lower()
    for k, v in cls.__dict__.items():
        if k == '__init__':
            arguments = []
            for an, av in v.__annotations__.items():
                if an != 'return':
                    arguments.append('+%s' % av.__name__)
            arguments.append('-term')

            def wrap(*args):
                return Constant(cls(*args))

            problog_export(*arguments)(wrap, funcname='%s_init' % (prefix))
        else:
            if type(v).__name__ == 'function':
                arguments = ['+term']
                for an, av in v.__annotations__.items():
                    if an == 'return':
                        arguments.append('-%s' % av.__name__)
                    else:
                        arguments.append('+%s' % av.__name__)

                problog_export(*arguments)(_call_func(v), funcname='%s_%s' % (prefix, k))


def _call_func(func):
    def wrap(s, *args):
        f = func(s.functor, *args)
        if f is None:
            return ()
        else:
            return f

    return wrap