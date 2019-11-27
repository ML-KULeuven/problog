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
import unittest

import glob, os, traceback, sys

from problog.forward import _ForwardSDD

if __name__ == '__main__' :
    sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from problog import root_path

from problog.program import PrologFile
from problog.tasks.dcproblog.parser import DCParser
from problog.tasks.dcproblog.solver import InferenceSolver

# noinspection PyBroadException
try:
    from pysdd import sdd
    has_sdd = True
except Exception as err:
    print("SDD library not available due to error: ", err, file=sys.stderr)
    has_sdd = False

try:
    import pyro
    has_pyro = True
except Exception as err:
    print("Pyro library not available due to error: ", err, file=sys.stderr)
    has_pyro = False

try:
    import psipy
    has_psi = True
except Exception as err:
    print("PyTorch library not available due to error: ", err, file=sys.stderr)
    has_psi = False


class TestDummy(unittest.TestCase):

    def test_dummy(self) : pass


class TestDCPyroGeneric(unittest.TestCase):

    def setUp(self) :
        try :
            self.assertSequenceEqual = self.assertItemsEqual
        except AttributeError :
            self.assertSequenceEqual = self.assertCountEqual

class TestDCPsiGeneric(unittest.TestCase):

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


def createDCPyroTestGeneric(filename, logspace=False) :

    correct = read_result(filename)

    def test(self):
        for eval_name in evaluatables:
            with self.subTest(evaluatable_name=eval_name, abe="pyro"):
                evaluate(self, evaluatable_name=eval_name)



        def evaluate(self, evaluatable_name=None) :
            try:
                abe = "pyro"
                args = {"device":"cpu", "ttype":"float64", "n_samples":50000}
                args["file_name"] = filename

                program = PrologFile(args['file_name'], parser=DCParser())
                solver = InferenceSolver(abe, **args)
                probabilities = solver.probability(program, **args)
                computed = {}
                for k,v in probabilities.items():
                    computed[str(k)] = float(v.value)

            except Exception as err :
                #print("exception %s" % err)
                e = err
                computed = None

            if computed is None :
                self.assertEqual(correct, type(e).__name__)
            else :
                self.assertIsInstance( correct, dict )
                self.assertSequenceEqual(correct, computed)

                for query in correct :
                    self.assertAlmostEqual(correct[query], computed[query], places=8, msg=query)

    return test


def createDCPsiTestGeneric(filename, logspace=False) :
        def test(self):
            for eval_name in evaluatables:
                with self.subTest(evaluatable_name=eval_name, abe="psi"):
                    evaluate(self, evaluatable_name=eval_name)

        return test




if __name__ == '__main__' :
    filenames = sys.argv[1:]
    pyro_filenames = [f for f in filenames if "pyro" in f]
    psi_filenames = [f for f in filenames if "psi" in f]
else :
    pyro_filenames= glob.glob( root_path('problog', 'tasks', 'dcproblog', 'test', 'pyro', '*.pl' ) )
    pyro_filenames += glob.glob( root_path('test', '*.pl' ) )
    psi_filenames= glob.glob( root_path('problog', 'tasks', 'dcproblog', 'test', 'psi', '*.pl' ) )
    print(pyro_filenames)

evaluatables = []

if has_sdd:
    evaluatables.append("sdd")
else:
    print("No SDD support - The DC tests are not performed with SDDs.")

if has_pyro:
    for testfile in pyro_filenames:
        testname = 'test_dc_pyro_' + os.path.splitext(os.path.basename(testfile))[0]
        setattr( TestDCPyroGeneric, testname, createDCPyroTestGeneric(testfile, True) )
else:
    print("No Pyro support - The DC tests are not performed with Pyro.")

if has_psi:
    for testfile in psi_filenames:
        testname = 'test_dc_psi_' + os.path.splitext(os.path.basename(testfile))[0]
        setattr( TestDCPsiGeneric, testname, createDCPsiTestGeneric(testfile, True) )
else:
    print("No PSI support - The DC tests are not performed with PSI.")

if __name__ == '__main__' :
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDCPyroGeneric)
    unittest.TextTestRunner(verbosity=2).run(suite)
