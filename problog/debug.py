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

    def __init__(self, keep_trace=True):
        self.call_redirect = {}
        self.call_results = defaultdict(int)
        self.level = 0    # level of calls (depth of stack)
        self.stack = []   # stack of calls (approximate?)
        if keep_trace:
            self.trace = []
        else:
            self.trace = None

        self.time_start = {}
        self.timestats = defaultdict(float)
        self.resultstats = defaultdict(int)
        self.callstats = defaultdict(int)
        self.time_start_global = None

    def process_message(self, msgtype, msgtarget, msgargs, context):
        if msgtarget is None and msgtype == 'r' and msgtarget in self.call_redirect:
            self.call_result(*(self.call_redirect[msgtarget] + (msgargs[0],)))

        if msgtype == 'r' and msgargs[3] and msgtarget in self.call_redirect:
            self.call_return(*self.call_redirect[msgtarget])
            del self.call_redirect[msgtarget]
        elif msgtype == 'c' and msgtarget in self.call_redirect:
            self.call_return(*self.call_redirect[msgtarget])
            del self.call_redirect[msgtarget]

    def call_create(self, node_id, functor, context, parent, location=None):
        if self.time_start_global is None:
            self.time_start_global = time.time()

        term = Term(functor, *context, location=location)
        if self.trace is not None:
            self.trace.append((self.level, "call", term, time.time()-self.time_start_global))
        self.time_start[term] = time.time(), location
        self.stack.append(term)
        self.level += 1
        self.call_redirect[parent] = (node_id, functor, context)

    def call_result(self, node_id, functor, context, result=None, location=None):
        term = Term(functor, *context, location=location)
        if self.trace is not None:
            self.trace.append((self.level, "result", term, time.time()-self.time_start_global, result))
        self.call_results[(node_id, term)] += 1

    def call_return(self, node_id, functor, context, location=None):
        term = Term(functor, *context, location=location)
        self.level -= 1
        ts, loc = self.time_start[term]

        now = time.time()
        if self.stack:
            self.stack.pop(-1)
        if self.trace is not None:
            if self.call_results[(node_id, term)] > 0:
                self.trace.append((self.level, "complete", term, now-self.time_start_global, now-ts))
            else:
                self.trace.append((self.level, "fail", term, now-self.time_start_global, now-ts))

        self.timestats[(term, loc)] += now - ts
        self.resultstats[(term, loc)] = self.call_results[node_id, term]
        self.callstats[(term, loc)] += 1
        # self.call_results[(node_id, term)] = 0

    def show_profile(self, aggregate=0):
        """Creates a table with profile information.

        :param aggregate: aggregation level (0: no aggregation, 1: same call, 2: predicate)
        :return: string
        """
        s = ''
        if aggregate == 0:
            s += '%50s\t %7s \t %4s \t %4s \t %s \n' % ("call", "time", "#sol", "#call", "location")
            s += '-' * 100 + '\n'
            for tm, key in sorted((t, k) for k, t in self.timestats.items()):
                term, location = key
                nb = self.resultstats[key]
                cl = self.callstats[key]
                location = location_string(location)
                s += '%50s\t %.5f \t %d \t %d \t [%s]\n' % (term, tm, nb, cl, location)
        else:
            timestats_agg = defaultdict(float)
            resultstats_agg = defaultdict(int)
            callstats_agg = defaultdict(int)

            for k, t in self.timestats.items():
                if aggregate == 1:
                    key = k[0]
                else:
                    key = k[0].signature
                timestats_agg[key] += t
                resultstats_agg[key] += self.resultstats[k]
                callstats_agg[key] += self.callstats[k]

            s += '%50s\t %7s \t %4s \t %4s\n' % ("call", "time", "#sol", "#call")
            s += '-' * 100 + '\n'
            for tm, key in sorted((t, k) for k, t in timestats_agg.items()):
                nb = resultstats_agg[key]
                cl = callstats_agg[key]
                s += '%50s\t %.5f \t %d \t %d\n' % (key, tm, nb, cl)

        return s

    def show_trace(self):
        s = ''
        if self.trace is not None:
            for record in self.trace:
                lvl = record[0]
                msg = record[1]
                term = record[2]
                tm_cumul = record[3]
                if msg == 'call':
                    s += "%s %s %s {%.5f} [%s]\n" % (' ' * lvl, msg, term, tm_cumul, location_string(term.location))
                elif msg == 'result':
                    args = record[4]
                    s += "%s %s %s %s {%.5f} [%s]\n" % (' ' * lvl, msg, term, args, tm_cumul, location_string(term.location))
                else:
                    tm_local = record[4]
                    s += "%s %s %s {%.5f} {%.5f} [%s]\n" % (' ' * lvl, msg, term, tm_cumul, tm_local, location_string(term.location))
        return s

def location_string(location):
    if location is None:
        return ''
    if type(location) == tuple:
        fn, ln, cn = location
        if fn is None:
            return 'at %s:%s' % (ln, cn)
        else:
            return 'at %s:%s in %s' % (ln, cn, fn)
    else:
        return 'at character %s' % location

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
