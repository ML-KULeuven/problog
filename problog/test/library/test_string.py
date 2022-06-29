"""
Part of the ProbLog distribution.

Copyright 2022 KU Leuven, DTAI Research Group

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
import unittest

from problog import get_evaluatable
from problog.program import PrologString


class TestLibString(unittest.TestCase):
    def test_str2lst(self):
        p = PrologString(
            """
                    :- use_module(library(string)).
                    test(L) :- str2lst('aaa zzz eee rrr', L).
                    query(test(L)).
                """
        )

        res = get_evaluatable().create_from(p).evaluate()
        self.assertEqual(list(res.values()), [1.0])  # probability check
        self.assertEqual(
            [str(x) for x in res.keys()],
            ["test([''', a, a, a, ' ', z, z, z, ' ', e, e, e, ' ', r, r, " "r, '''])"],
        )  # name check

    def test_lst2str(self):
        p = PrologString(
            """
                    :- use_module(library(string)).
                    test(L) :- lst2str(['aaa','dddd','fffff'], L).
                    query(test(L)).
                """
        )

        res = get_evaluatable().create_from(p).evaluate()
        self.assertEqual(list(res.values()), [1.0])  # probability check
        self.assertEqual(
            [str(x) for x in res.keys()], ["test(aaaddddfffff)"]
        )  # name check

    def test_join(self):
        p = PrologString(
            """
                    :- use_module(library(string)).
                    test(L) :- join(',', ['aaa','dddd','fffff'], L).
                    query(test(L)).
                """
        )

        res = get_evaluatable().create_from(p).evaluate()
        self.assertEqual(list(res.values()), [1])  # probability check
        self.assertEqual(
            [str(x) for x in res.keys()], ["test('aaa,dddd,fffff')"]
        )  # name check

    def test_concat(self):
        p = PrologString(
            """
                    :- use_module(library(string)).
                    test(L) :- concat(['aaa','dddd','fffff'], L).
                    query(test(L)).
                """
        )
        res = get_evaluatable().create_from(p).evaluate()
        self.assertEqual(list(res.values()), [1.0])  # probability check
        self.assertEqual(
            [str(x) for x in res.keys()], ["test(aaaddddfffff)"]
        )  # name check
