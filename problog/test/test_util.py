import os
import unittest

from problog.logic import Constant, Term, Not, Var
from problog.tasks import map, explain, time1, bayesnet, mpe, ground, probability
from problog.clausedb import TermTrieNode
#TODO: move to util


class TestUtil(unittest.TestCase):

    # BN
    def test_check_bn(self):
        # term0 tests the initialisation (and Var)
        # term1 tests is_leaf() case
        # term2 tests non-matching due to functor
        # term3 tests non-matching due to arity
        # term4 tests Var

        # Items to be inserted into Trie
        term0 = Term("publication", Term("type2", Constant(1)), Var('X'))  # publication(type2(1), X)
        term1 = Term("publication", Term("type2", Constant(1)), Constant(2))  # publication(type2(1), 2)
        term2 = Term("publication", Term("type2", Constant(2)), Constant(2))  # publication(type2(2), 2)
        term3 = Term("publication", Term("type2", Constant(3)))  # publication(type2(3))
        term4 = Term("publication", Term("type2", Var('X')))  # publication(type2(X))
        term5 = Term("publication", Var('X'), Constant(2))  # publication(X,2)
        terms = [term0, term1, term2, term3, term4, term5]
        # Items that should match with each of the terms
        matching_items = [
            {0, 1, 5},
            {0, 1, 5},
            {2, 5},
            {3, 4},
            {3, 4},
            {0, 1, 2, 5}
        ]
        # Insert into Trie
        trie = TermTrieNode(functor=None, arity=None, arg_stack=None, item=None)
        for index, term in enumerate(terms):
            trie.add_term(term, index)
        print(trie)

        # Check matchings
        for index, term in enumerate(terms):
            found_items = trie.find(term)
            print(f"found items {index}: {found_items}")
            assert matching_items[index] == found_items

        #TODO add test on adding to intermediate node where only Var children are present


if __name__ == '__main__':
    unittest.main()