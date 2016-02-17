"""
Module name
"""

from __future__ import print_function

import sys


class ProbLogError(Exception):
    """General Problog error. Root of all ProbLog errors that can be blamed on user input.

    :param message: error message
    :param location: location at which the error occurred (referring to an user input file)
    """

    def __init__(self, message, location=None, **extra):
        self.errtype = self.__class__.__name__
        self.base_message = message
        self.location = location
        self.message = self._message()
        for k, v in extra.items():
            setattr(self, k, v)

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
        return '%s%s.' % (self.base_message, self._location_string())

    def __str__(self):
        return self.message


class ParseError(ProbLogError):
    """Error during parsing."""
    pass


class GroundingError(ProbLogError):
    """Represents an error that occurred during grounding."""

    pass


class CompilationError(ProbLogError):
    """Error during compilation"""
    pass


class InstallError(ProbLogError):
    """Error during installation"""
    pass


class InvalidValue(ProbLogError):
    pass


class UserError(ProbLogError):
    pass


class InconsistentEvidenceError(ProbLogError):
    """Error when evidence is inconsistent"""

    def __init__(self, source=None, context=''):
        """

        :param source: evidence term that causes the problem
        :param context: extra message describing the context (e.g. example number in lfi)
        :return:
        """
        self.source = source
        self.context = context
        if source is None:
            ProbLogError.__init__(self, "Inconsistent evidence detected%s" % context)
        else:
            ProbLogError.__init__(self, "Inconsistent evidence detected%s: '%s'" % (context, source))


def process_error(err, debug=False):
    """Take the given error raise by ProbLog and produce a meaningful error message.

    :param err: error that was raised
    :param debug: if True, also print original stack trace
    :return: textual representation of the error
    """
    if debug and hasattr(err, 'trace'):
        print(err.trace, file=sys.stderr)

    if isinstance(err, ProbLogError):
        return '%s: %s' % (err.__class__.__name__, err)
    else:
        if not debug and hasattr(err, 'trace'):
            print(err.trace, file=sys.stderr)
        return 'An unexpected error has occurred.'
