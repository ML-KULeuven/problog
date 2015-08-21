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
    def __init__(self):
        self.call_redirect = {}
        self.call_results = defaultdict(int)
        self.level = 0

    def process_message(self, msgtype, msgtarget, msgargs, context):
        if msgtype == 'r' and msgargs[3] and msgtarget in self.call_redirect:
            self.call_return(*self.call_redirect[msgtarget])
        elif msgtype == 'c' and msgtarget in self.call_redirect:
            self.call_return(*self.call_redirect[msgtarget])

    def call_create(self, node_id, functor, context, parent):
        print('  ' * self.level, "call", Term(functor, *context))
        self.level += 1
        self.call_redirect[parent] = (node_id, functor, context)

    def call_result(self, node_id, functor, context, result):
        print('  ' * self.level, "result", Term(functor, *result))
        self.call_results[(node_id, functor, tuple(context))] += 1

    def call_return(self, node_id, functor, context):
        self.level -= 1
        if self.call_results[(node_id, functor, tuple(context))] > 0:
            # print ('  ' * self.level, "return success")
            print('  ' * self.level, "complete", Term(functor, *context))
        else:
            print('  ' * self.level, "fail", Term(functor, *context))
        self.call_results[(node_id, functor, tuple(context))] = 0

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
