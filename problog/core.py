"""
problog.core - Binary Decision Diagrams
----------------------------------------------

Provides core functionality of ProbLog.

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

from collections import defaultdict

import traceback


class ProbLog(object):
    """Static class containing transformation information"""

    def __init__(self):
        raise RuntimeError("This is a static class!")

    transformations = defaultdict(list)
    create_as = defaultdict(list)

    @classmethod
    def register_transformation(cls, src, target, action=None):
        """Register a transformation from class src to class target using function action.

        :param src: source function
        :param target: target function
        :param action: transformation function
        """
        cls.transformations[target].append((src, action))

    @classmethod
    def register_create_as(cls, repl, orig):
        """Register that we can create objects of class `repl` in the same way as objects \
        of class `orig`.

        :param repl: object we want to create
        :param orig: object construction we can use instead
        """
        cls.create_as[repl].append(orig)

    @classmethod
    def find_paths(cls, src, target, stack=()):
        """Find all possible paths to transform the src object into the target class.

        :param src: object to transform
        :param target: class to tranform the object to
        :param stack: stack of intermediate classes
        :return: list of class, action, class, action, ..., class
        """
        # Create a destination object or any of its subclasses
        if isinstance(src, target):
            yield (target,)
        else:
            # for d in list(cls.transformations) :
            #     if issubclass(d,target) :
            targets = [target] + cls.create_as[target]
            for d in targets:
                for s, action in cls.transformations[d]:
                    if s not in stack:
                        for path in cls.find_paths(src, s, stack + (s,)):
                            yield path + (action, target)

    @classmethod
    def convert(cls, src, target, **kwdargs):
        """Convert the source object into an object of the target class.

        :param src: source object
        :param target: target class
        :param kwdargs: additional arguments passed to transformation functions
        """
        # Find transformation paths from source to target.
        for path in cls.find_paths(src, target):
            try:
                # A path is a sequence of obj, function, obj/class, ..., obj/class
                current_obj = src
                path = path[1:]
                # Invariant: path[0] is a function, path[1] is an obj/class
                while path:
                    if path[1] is not None:
                        next_obj = path[0](current_obj, path[1](**kwdargs), **kwdargs)
                    else:
                        next_obj = path[1].create_from_default_action(current_obj, **kwdargs)
                    path = path[2:]
                    current_obj = next_obj
                return current_obj
            except TransformationUnavailable:
                # The current transformation strategy didn't work for some reason. Try another one.
                pass
        raise ProbLogError("No conversion strategy found from an object of "
                           "class '%s' to an object of class '%s'."
                           % (type(src).__name__, getattr(target, '__name__')))


class ProbLogError(Exception):
    """General Problog error. Root of all user-caused ProbLog errors."""
    pass


class TransformationUnavailable(Exception):
    """Exception thrown when no valid transformation between two ProbLogObjects can be found."""
    pass


class ParseError(ProbLogError):
    """Error during parsing."""
    def __init__(self, message, lineno, col, line):
        self.lineno = lineno
        self.col = col
        self.msg = message
        self.line = line
        Exception.__init__(self, '%s (at %s:%s)' % (self.msg, self.lineno, self.col))


class GroundingError(ProbLogError):
    """Represents an error that occurred during grounding."""

    def __init__(self, base_message, location=None):
        self.base_message = base_message
        self.location = location

    def _location_string(self):
        if self.location is None:
            return ''
        if type(self.location) == tuple:
            fn, ln, cn = self.location
            if fn is None:
                return ' at %s:%s' % (ln, cn)
            else:
                return ' at %s:%s in %s' % (ln, cn, fn)
        else:
            return ' at character %s' % self.location

    def _message(self):
        return '%s: %s%s.' % (self.__class__.__name__, self.base_message, self._location_string())

    def __str__(self):
        return self._message()


class CompilationError(ProbLogError):
    """Error during compilation"""
    pass


class InstallError(ProbLogError):
    """Error during installation"""
    pass


class InconsistentEvidenceError(ProbLogError):
    """Error when evidence is inconsistent"""

    def __init__(self, message=None):
        if message is None:
            message = 'The given evidence is inconsistent.'
        ProbLogError.__init__(self, message)


def process_error(err, debug=False):
    """Take the given error raise by ProbLog and produce a meaningful error message.

    :param err: error that was raised
    :param debug: if True, also print original stack trace
    :return: textual representation of the error
    """
    if debug:
        traceback.print_exc()

    err_type = type(err).__name__
    if err_type == 'ParseException':
        return 'Parsing error on %s:%s: %s.\n%s' % (err.lineno, err.col, err.msg, err.line)
    elif isinstance(err, ParseError):
        return 'Parsing error on %s:%s: %s.\n%s' % (err.lineno, err.col, err.msg, err.line)
    elif isinstance(err, GroundingError):
        return 'Error during grounding: %s' % err
    elif isinstance(err, CompilationError):
        return 'Error during compilation: %s' % err
    elif isinstance(err, ProbLogError):
        return 'Error: %s' % err
    else:
        if not debug:
            traceback.print_exc()
        return 'Unknown error: %s' % err_type


class ProbLogObject(object):
    """Root class for all convertible objects in the ProbLog system."""

    @classmethod
    def create_from(cls, obj, **kwdargs):
        """Transform the given object into an object of the current class using transformations.

        :param obj: obj to transform
        :param kwdargs: additional options
        :return: object of current class
        """
        return ProbLog.convert(obj, cls, **kwdargs)

    # noinspection PyPep8Naming
    @classmethod
    def createFrom(cls, obj, **kwdargs):
        """Transform the given object into an object of the current class using transformations.

        :param obj: obj to transform
        :param kwdargs: additional options
        :return: object of current class
        """
        return cls.create_from(obj, **kwdargs)

    @classmethod
    def create_from_default_action(cls, src):
        """Create object of this class from given source object using default action.

        :param src: source object to transform
        :return: transformed object
        """

        raise ProbLogError("No default conversion strategy defined.")


def transform_create_as(cls1, cls2):
    """Informs the system that cls1 can be used instead of cls2 in any transformations.

    :param cls1:
    :param cls2:
    :return:
    """
    ProbLog.register_create_as(cls1, cls2)


# noinspection PyPep8Naming
class transform(object):
    """Decorator for registering a transformation between two classes.

    :param cls1: source class
    :param cls2: target class
    :param func: transformation function (for direct use instead of decorator)
    """
    def __init__(self, cls1, cls2, func=None):
        self.cls1 = cls1
        self.cls2 = cls2
        if not issubclass(cls2, ProbLogObject):
            raise TypeError("Conversion only possible for subclasses of ProbLogObject.")
        if func is not None:
            self(func)

    def __call__(self, f):
        ProbLog.register_transformation(self.cls1, self.cls2, f)
        return f


def list_transformations():
    """Print an overview of available transformations."""
    print ('Available transformations:')
    for target in ProbLog.transformations:
        print ('\tcreate %s.%s' % (target.__module__, target.__name__))
        for src, func in ProbLog.transformations[target]:
            print ('\t\tfrom %s.%s by %s.%s' %
                   (src.__module__, src.__name__, func.__module__, func.__name__))
