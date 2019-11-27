import unittest

import glob, os, traceback, sys


from problog.program import PrologFile
from problog.tasks.dcproblog.parser import DCParser
from problog.tasks.dcproblog.solver import InferenceSolver

if __name__ == "__main__":
    sys.path.insert(
        0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    )


from problog import root_path
test_base = root_path('problog', 'tasks', 'dcproblog', 'test')



def get_expected(filename) :
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


def get_filenames(*args):
    path2files = os.path.join(test_base, *args)
    testfiles = [os.path.join(path2files,f) for f in os.listdir(path2files) if os.path.isfile(os.path.join(path2files, f))]
    return testfiles


class TestTasks(unittest.TestCase):
    def test_pyro_examples(self):
        testfiles = get_filenames("pyro", "examples")
        abe = "pyro"
        args = {"device":"cpu", "ttype":"float64", "n_samples":50000}

        for tf in testfiles:
            expected = get_expected(tf)

            args["file_name"] = tf
            program = PrologFile(args['file_name'], parser=DCParser())
            solver = InferenceSolver(abe, **args)
            probabilities = solver.probability(program, **args)
            computed = {}
            for k,v in probabilities.items():
                computed[str(k)] = float(v.value)

            for query in expected :
                self.assertAlmostEqual(expected[query], computed[query], places=2, msg=query)


    # def test_psi_examples(self):
    #     pass

    # def test_problog_system(self):
    #     path2files = root_path("test")
    #     testfiles = [os.path.join(path2files,f) for f in os.listdir(path2files) if os.path.isfile(os.path.join(path2files, f))]
    #
    #     abe = "pyro"
    #     args = {"device":"cpu", "ttype":"float64", "n_samples":1000}
    #
    #     for tf in testfiles:
    #         expected = get_expected(tf)
    #
    #         args["file_name"] = tf
    #         program = PrologFile(args['file_name'], parser=DCParser())
    #         solver = InferenceSolver(abe, **args)
    #         probabilities = solver.probability(program, **args)
    #         computed = {}
    #         for k,v in probabilities.items():
    #             computed[str(k)] = float(v.value)
    #
    #         for query in expected :
    #             self.assertAlmostEqual(expected[query], computed[query], places=5, msg=query)


if __name__ == "__main__":
    unittest.main()
