import unittest

from problog import root_path

from problog.setup import install
from problog.program import PrologFile
from problog.nnf_formula import NNF
from problog.sdd_formula import SDD

import glob, os

class TestDummy(unittest.TestCase):
    
    def test_dummy(self) : pass

class TestSystemSDD(unittest.TestCase) :

    def setUp(self) :

        install()

        try :
            self.assertSequenceEqual = self.assertItemsEqual
        except AttributeError :
            self.assertSequenceEqual = self.assertCountEqual

class TestSystemNNF(unittest.TestCase) :

    def setUp(self) :

        try :
            self.assertSequenceEqual = self.assertItemsEqual
        except AttributeError :
            self.assertSequenceEqual = self.assertCountEqual



def read_result(filename) :
    results = {}
    with open( filename ) as f :
        reading = False
        for l in f :
            if l.strip().startswith('%Expected outcome:') :
                reading = True
            elif reading :
                if l.strip().startswith('% ') :
                    query, prob = l[2:].rsplit(None,1)
                    results[query.strip()] = float(prob.strip())
                else :
                    break
    return results

def createSystemTestSDD(filename) :

    correct = read_result(filename)

    def test(self) :

        sdd = SDD.createFrom(PrologFile(filename))
        computed = sdd.evaluate()

        self.assertSequenceEqual(correct, computed)

        for query in correct :
            self.assertAlmostEqual(correct[query], computed[query])

    return test


def createSystemTestNNF(filename) :

    correct = read_result(filename)

    def test(self) :

        sdd = NNF.createFrom(PrologFile(filename))
        computed = sdd.evaluate()

        self.assertSequenceEqual(correct, computed)

        for query in correct :
            self.assertAlmostEqual(correct[query], computed[query])

    return test


for testfile in glob.glob( root_path('test', '*.pl' ) ) :
    testname = 'test_' + os.path.basename(testfile)
    setattr( TestSystemSDD, testname, createSystemTestSDD(testfile) )
    setattr( TestSystemNNF, testname, createSystemTestNNF(testfile) )


#
# TestSystem.testB = createSystemTestSDD(root_path('test', '2_tossing_coin.pl' ) )