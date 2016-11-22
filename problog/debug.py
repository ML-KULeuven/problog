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

from .logic import Term
from collections import defaultdict
import time


def printtrace(func):
    def _wrapped_function(*args, **kwdargs):
        print('CALL:', func.__name__, args, kwdargs)
        result = func(*args, **kwdargs)
        print('RETURN:', func.__name__, args, kwdargs, result)
        return result

    return _wrapped_function


def printtrace_self(func):
    def _wrapped_function(self, *args, **kwdargs):
        print('CALL:', func.__name__, args, kwdargs)
        result = func(self, *args, **kwdargs)
        print('RETURN:', func.__name__, args, kwdargs, result)
        return result

    return _wrapped_function


class EngineTracer(object):

    def __init__(self, keep_trace=False):
        self.call_redirect = {}
        self.call_results = defaultdict(int)
        self.level = 0    # level of calls (depth of stack)
        self.stack = []   # stack of calls (approximate?)
        if keep_trace:
            self.trace = []
        else:
            self.trace = None

        self.time_start = {}
        self.timestats = {}

    def process_message(self, msgtype, msgtarget, msgargs, context):
        if msgtarget is None and msgtype == 'r' and msgtarget in self.call_redirect:
            self.call_result(*(self.call_redirect[msgtarget] + msgargs[0]))

        if msgtype == 'r' and msgargs[3] and msgtarget in self.call_redirect:
            self.call_return(*self.call_redirect[msgtarget])
            del self.call_redirect[msgtarget]
        elif msgtype == 'c' and msgtarget in self.call_redirect:
            self.call_return(*self.call_redirect[msgtarget])
            del self.call_redirect[msgtarget]

    def call_create(self, node_id, functor, context, parent, location=None):
        term = Term(functor, *context, location=location)
        if self.trace is not None:
            self.trace.append((self.level, "call", term))
        self.time_start[term] = time.time(), location
        self.stack.append(term)
        self.level += 1
        self.call_redirect[parent] = (node_id, functor, context)

    def call_result(self, node_id, functor, context, result, location=None):
        term = Term(functor, *context, location=location)
        if self.trace is not None:
            self.trace.append((self.level, "result", term))
        self.call_results[(node_id, term)] += 1

    def call_return(self, node_id, functor, context, location=None):
        term = Term(functor, *context, location=location)
        self.level -= 1
        if self.stack:
            self.stack.pop(-1)
        if self.trace is not None:
            if self.call_results[(node_id, term)] > 0:
                self.trace.append((self.level, "complete", term))
            else:
                self.trace.append((self.level, "fail", term))

        ts, loc = self.time_start[term]
        self.timestats[(term, loc)] = (time.time() - ts, self.call_results[node_id, term])
        # self.call_results[(node_id, term)] = 0

    def show_profile(self):
        s = ''
        s += '%37s\t %7s \t %4s \t %s \n' % ("call", "time", "#sol", "location")
        s += '-' * 80 + '\n'
        for tm, key in sorted((t, k) for k, t in self.timestats.items()):
            term, location = key
            tm, nb = tm
            location = location_string(location)
            s += '%37s\t %.5f \t %d \t [%s]\n' % (term, tm, nb, location)
        return s

    def show_trace(self):
        s = ''
        if self.trace is not None:
            for lvl, msg, term in self.trace:
                s += "%s %s %s\n" % (' ' * lvl, msg, term)
        return s

def location_string(location):
    if location is None:
        return ''
    if type(location) == tuple:
        fn, ln, cn = location
        if fn is None:
            return ' at %s:%s' % (ln, cn)
        else:
            return ' at %s:%s in %s' % (ln, cn, fn)
    else:
        return ' at character %s' % location

# Assume the program
#   a(1).
#   a(2).
#   p(X,Y) :- a(X), a(Y), X \= Y.
#   p(1,1).

# Call stack:
#   -> p(X,Y) == [1] a(X), [2] a(Y), [3] X \= Y
#       [1] a(X), a(Y), X \= Y
#           -> a(X)
#           <- a(1) [FACT]
#           <- a(2) [FACT]
#       [2] a(Y), 1\=X
#           -> a(Y) [CACHE]
#           <- a(1)
#           <- a(2)
#       [3] 1 \= 1
#           -> 1 \= 1
#           <- FAIL
#       [3] 1 \= 2
#           -> 1 \= 2
#           <- TRUE
#       [2] a(Y), 2 \= Y
#           -> a(Y) [CACHE]
#           <- a(1)
#           <- a(2)
#       [3] 2 \= 1
#           -> 2 \= 1
#           <- TRUE
#       [3] 2 \= 2
#           -> 2 \= 2
#           <- FAIL
#   <- p(1,2)
#   <- p(2,1)
#   <- p(1,1) [FACT]
