import unittest

import glob, os, traceback, sys

if __name__ == '__main__' :
    sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from problog import root_path

from problog.setup import install
from problog.program import PrologFile, DefaultPrologParser, ExtendedPrologFactory
from problog.nnf_formula import NNF
from problog.sdd_formula import SDD

class TestDummy(unittest.TestCase):
    
    def test_dummy(self) : pass

class TestSystemSDD(unittest.TestCase) :

    def setUp(self) :

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
            l = l.strip()
            if l.startswith('%Expected outcome:') :
                reading = True
            elif reading :
                if l.lower().startswith('% error') :
                    return l[len('% error'):].strip()
                elif l.startswith('% ') :
                    query, prob = l[2:].rsplit(None,1)
                    results[query.strip()] = float(prob.strip())
                else :
                    reading = False
            if l.startswith('query(') and l.find('% outcome:') >= 0 :
                pos = l.find('% outcome:')
                query = l[6:pos].strip().rstrip('.').rstrip()[:-1]
                prob = l[pos+10:]
                results[query.strip()] = float(prob.strip())
    return results

def createSystemTestSDD(filename) :

    correct = read_result(filename)

    def test(self) :
        try :
            parser = DefaultPrologParser(ExtendedPrologFactory())
            sdd = SDD.createFrom(PrologFile(filename, parser=parser))
            computed = sdd.evaluate()
        except Exception as err :
            e = err
            computed = None
            
        if computed == None :
            self.assertEqual(correct, type(e).__name__)
        else :
            self.assertIsInstance( correct, dict )
            self.assertSequenceEqual(correct, computed)

            for query in correct :
                self.assertAlmostEqual(correct[query], computed[query], msg=query)


    return test


def createSystemTestNNF(filename) :

    correct = read_result(filename)

    def test(self) :
        try :
            parser = DefaultPrologParser(ExtendedPrologFactory())
            sdd = NNF.createFrom(PrologFile(filename, parser=parser))
            computed = sdd.evaluate()
        except Exception as err :
            e = err
            computed = None
            
        if computed == None :
            self.assertEqual(correct, type(e).__name__)
        else :
            self.assertIsInstance( correct, dict )
            self.assertSequenceEqual(correct, computed)

            for query in correct :
                self.assertAlmostEqual(correct[query], computed[query], msg=query)

    return test


if __name__ == '__main__' :
    filenames = sys.argv[1:]    
else :
    filenames = glob.glob( root_path('test', '*.pl' ) )

for testfile in filenames :
    testname = 'test_system_' + os.path.splitext(os.path.basename(testfile))[0]
    setattr( TestSystemSDD, testname, createSystemTestSDD(testfile) )
    setattr( TestSystemNNF, testname, createSystemTestNNF(testfile) )
    
if __name__ == '__main__' :
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSystemSDD)
    unittest.TextTestRunner(verbosity=2).run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSystemNNF)
    unittest.TextTestRunner(verbosity=2).run(suite)