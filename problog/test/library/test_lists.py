"""
Part of the ProbLog distribution.

Copyright 2026 KU Leuven, DTAI Research Group

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
import re
import unittest

from problog import get_evaluatable
from problog.program import PrologString


def evaluate(program):
    result = get_evaluatable().create_from(PrologString(program)).evaluate()
    return {str(k): v for k, v in result.items()}


class TestLibLists(unittest.TestCase):
    def test_memberchk(self):
        """A ground list takes the Python path in lists.py."""
        res = evaluate(
            """
            :- use_module(library(lists)).
            hit             :- memberchk(2, [1,2,3]).
            miss            :- memberchk(9, [1,2,3]).
            empty           :- memberchk(a, []).
            binds_element   :- memberchk(f(X), [g(1), f(7), f(8)]), X =:= 7.
            takes_the_first :- memberchk(X, [4,5]), X =:= 4.
            nested          :- memberchk([1,2], [[3],[1,2]]).
            not_a_list      :- memberchk(a, notalist).
            improper_list   :- memberchk(a, [x,y|z]).
            query(hit). query(miss). query(empty). query(binds_element).
            query(takes_the_first). query(nested). query(not_a_list).
            query(improper_list).
        """
        )
        self.assertEqual(
            {
                "hit": 1.0,
                "miss": 0.0,
                "empty": 0.0,
                "binds_element": 1.0,
                "takes_the_first": 1.0,
                "nested": 1.0,
                "not_a_list": 0.0,
                "improper_list": 0.0,
            },
            res,
        )

    def test_memberchk_non_ground(self):
        """A list that is not ground falls back on the clauses in lists.pl.

        The bindings it makes are the reason it has to: they reach into the
        list, and into a variable shared between both arguments.
        """
        res = evaluate(
            """
            :- use_module(library(lists)).
            open_list(L)   :- memberchk(a, L).
            open_tail(T)   :- memberchk(a, [b|T]).
            partial(L)     :- memberchk(a, [x,y|_]), L = done.
            in_the_list(X) :- memberchk(a, [X,b]).
            shared(X, E)   :- memberchk(X, [E,b]).
            query(open_list(_)). query(open_tail(_)). query(partial(_)).
            query(in_the_list(_)). query(shared(_,_)).
        """
        )
        self.assertEqual([1.0] * 5, sorted(res.values()))
        # The numbering of the free variables depends on how many the program
        # has used before, so compare the shape of the answers.
        self.assertEqual(
            [
                "in_the_list(a)",
                "open_list([a | _])",
                "open_tail([a | _])",
                "partial(done)",
                "shared(_,_)",
            ],
            sorted(re.sub(r"X\d+", "_", name) for name in res),
        )
        # memberchk(X, [E,b]) unifies X and E with each other, so the two
        # arguments have to come back as the same variable.
        shared = [name for name in res if name.startswith("shared(")][0]
        self.assertEqual(*shared[len("shared(") : -1].split(","))
