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
from .logic import Constant, term2list, list2term, Term, Object
from .engine_builtin import check_mode, is_variable
from .engine_unify import UnifyError, unify_value
import inspect
import os


# noinspection PyPep8Naming
class problog_export(object):
    database = None

    @classmethod
    def add_function(cls, name, in_args, out_args, func, module_name):
        if problog_export.database is not None:
            problog_export.database.add_extern(
                name, in_args + out_args, func, scope=module_name
            )

    # noinspection PyUnusedLocal
    def __init__(self, *args, **kwdargs):
        # TODO check if arguments are in order: input first, output last
        self.input_arguments = [a[1:] for a in args if a[0] == "+"]
        self.output_arguments = [a[1:] for a in args if a[0] == "-"]
        self.functor = kwdargs.get("functor")

    def _convert_input(self, a, t):
        if t == "str":
            return str(a)
        elif t == "int":
            return int(a)
        elif t == "float":
            return float(a)
        elif t == "list":
            return term2list(a)
        elif t == "term":
            return a
        elif t == "obj":
            return a.functor
        else:
            raise ValueError("Unknown type specifier '%s'!" % t)

    def _type_to_callmode(self, t):
        if t == "str":
            return "a"
        elif t == "int":
            return "i"
        elif t == "float":
            return "f"
        elif t == "list":
            return "L"
        elif t == "term":
            return "*"
        elif t == "obj":
            return "o"
        else:
            raise ValueError("Unknown type specifier '%s'!" % t)

    def _extract_callmode(self):
        callmode_in = ""
        for t in self.input_arguments:
            callmode_in += self._type_to_callmode(t)

        # multiple call modes: index = binary encoding on whether the output is bound
        # 0 -> all unbound
        # 1 -> first output arg is bound
        # 2 -> second output arg is bound
        # 3 -> first and second are bound

        n = len(self.output_arguments)
        for i in range(0, 1 << n):
            callmode = callmode_in
            for j, t in enumerate(self.output_arguments):
                if i & (1 << (n - j - 1)):
                    callmode += self._type_to_callmode(t)
                else:
                    callmode += "v"
            yield callmode

    def _convert_output(self, a, t):
        if t == "str":
            return Term(a)
        elif t == "int":
            return Constant(a)
        elif t == "float":
            return Constant(a)
        elif t == "list":
            return list2term(a)
        elif t == "term":
            if not isinstance(a, Term):
                return Term(a)
                # raise ValueError("Expected term output, got '%s' instead." % type(a))
            return a
        elif t == "obj":
            if not isinstance(a, Object):
                return Object(a)
            else:
                return a
        else:
            raise ValueError("Unknown type specifier '%s'!" % t)

    def _convert_inputs(self, args):
        return [self._convert_input(a, t) for a, t in zip(args, self.input_arguments)]

    def _convert_outputs(self, args):
        return [self._convert_output(a, t) for a, t in zip(args, self.output_arguments)]

    def __call__(self, func, funcname=None, modname="#auto#"):
        if funcname is None:
            if self.functor is None:
                funcname = func.__name__
            else:
                funcname = self.functor
        if modname == "#auto#":
            modname = func.__module__

        def _wrapped_function(*args, **kwdargs):
            bound = check_mode(
                args, list(self._extract_callmode()), funcname, **kwdargs
            )
            converted_args = self._convert_inputs(args)

            argspec = inspect.getargspec(func)
            if argspec.keywords is not None:
                result = func(*converted_args, **kwdargs)
            else:
                result = func(*converted_args)
            if len(self.output_arguments) == 1:
                result = [result]

            try:
                transformed = []
                for i, r in enumerate(result):
                    r = self._convert_output(r, self.output_arguments[i])
                    if bound & (1 << (len(self.output_arguments) - i - 1)):
                        r = unify_value(r, args[len(self.input_arguments) + i], {})
                    transformed.append(r)
                result = args[: len(self.input_arguments)] + tuple(transformed)
                return [result]
            except UnifyError:
                return []

        problog_export.add_function(
            funcname,
            len(self.input_arguments),
            len(self.output_arguments),
            _wrapped_function,
            module_name=modname,
        )
        return func


# noinspection PyPep8Naming
class problog_export_raw(problog_export):

    # noinspection PyUnusedLocal
    def __init__(self, *args, **kwdargs):
        problog_export.__init__(self, *args, **kwdargs)
        self.input_arguments = [a[1:] for a in args]

    def _convert_input(self, a, t):
        if is_variable(a):
            return None
        else:
            return problog_export._convert_input(self, a, t)

    def _extract_callmode(self):
        callmode_in = ""

        # multiple call modes: index = binary encoding on whether the output is bound
        # 0 -> all unbound
        # 1 -> first output arg is bound
        # 2 -> second output arg is bound
        # 3 -> first and second are bound

        n = len(self.input_arguments)
        for i in range(0, 1 << n):
            callmode = callmode_in
            for j, t in enumerate(self.input_arguments):
                if i & (1 << (n - j - 1)):
                    callmode += self._type_to_callmode(t)
                else:
                    callmode += "v"
            yield callmode

    def __call__(self, func, funcname=None, modname="#auto#"):
        if funcname is None:
            if self.functor is None:
                funcname = func.__name__
            else:
                funcname = self.functor

        if modname == "#auto#":
            modname = func.__module__

        def _wrapped_function(*args, **kwdargs):
            bound = check_mode(
                args, list(self._extract_callmode()), funcname, **kwdargs
            )
            converted_args = self._convert_inputs(args)
            results = []
            for result in func(*converted_args, **kwdargs):
                if len(result) == 2 and type(result[0]) == tuple:
                    # Probabilistic
                    result, p = result
                    raise Exception("We don't support probabilistic yet!")
                else:
                    p = None

                # result is always a list of tuples
                try:
                    transformed = []
                    for i, r in enumerate(result):
                        r = self._convert_output(r, self.input_arguments[i])
                        if bound & (1 << i):
                            r = unify_value(r, args[i], {})
                        transformed.append(r)
                    from .engine_stack import Context, get_state

                    result = Context(tuple(transformed), state=get_state(result))
                    results.append(result)
                except UnifyError:
                    pass
            return results

        problog_export.add_function(
            funcname,
            len(self.input_arguments),
            0,
            _wrapped_function,
            module_name=modname,
        )
        return func


# noinspection PyPep8Naming
class problog_export_nondet(problog_export):
    def __call__(self, func, funcname=None, modname="#auto#"):
        if funcname is None:
            if self.functor is None:
                funcname = func.__name__
            else:
                funcname = self.functor
        if modname == "#auto#":
            modname = func.__module__

        def _wrapped_function(*args, **kwdargs):
            bound = check_mode(
                args, list(self._extract_callmode()), funcname, **kwdargs
            )
            converted_args = self._convert_inputs(args)
            results = []
            argspec = inspect.getargspec(func)
            if argspec.keywords is not None:
                func_result = func(*converted_args, **kwdargs)
            else:
                func_result = func(*converted_args)
            for result in func_result:
                if len(self.output_arguments) == 1:
                    result = [result]

                try:
                    transformed = []
                    for i, r in enumerate(result):
                        r = self._convert_output(r, self.output_arguments[i])
                        if bound & (1 << (len(self.output_arguments) - i - 1)):
                            r = unify_value(r, args[len(self.input_arguments) + i], {})
                        transformed.append(r)
                    result = args[: len(self.input_arguments)] + tuple(transformed)
                    results.append(result)
                except UnifyError:
                    pass
            return results

        problog_export.add_function(
            funcname,
            len(self.input_arguments),
            len(self.output_arguments),
            _wrapped_function,
            module_name=modname,
        )
        return func


def problog_export_class(cls):
    prefix = cls.__name__.lower()
    for k, v in cls.__dict__.items():
        if k == "__init__":
            arguments = []
            for an, av in v.__annotations__.items():
                if an != "return":
                    arguments.append("+%s" % av.__name__)
            arguments.append("-term")

            def wrap(*args):
                return Constant(cls(*args))

            problog_export(*arguments)(wrap, funcname="%s_init" % (prefix))
        else:
            if type(v).__name__ == "function":
                arguments = ["+term"]
                for an, av in v.__annotations__.items():
                    if an == "return":
                        arguments.append("-%s" % av.__name__)
                    else:
                        arguments.append("+%s" % av.__name__)

                problog_export(*arguments)(
                    _call_func(v), funcname="%s_%s" % (prefix, k)
                )


def _call_func(func):
    def wrap(s, *args):
        f = func(s.functor, *args)
        if f is None:
            return ()
        else:
            return f

    return wrap
