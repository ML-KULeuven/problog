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
                    results[query.strip()] = prob.strip()
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
            expected = {}
            expected = get_expected(tf)

            args["file_name"] = tf
            program = PrologFile(args['file_name'], parser=DCParser())
            solver = InferenceSolver(abe, **args)

            probability = {}
            probabilities = solver.probability(program, **args)
            computed = {}
            for k,v in probabilities.items():

                computed[str(k)] = float(v.value)

            for query in expected :
                self.assertAlmostEqual(float(expected[query]), computed[query], places=2, msg=query)



    def test_psi_examples(self):
        try:
            import psipy
        except ImportError:
            return
        testfiles = get_filenames("psi", "examples")
        abe = "psi"
        args = {}
        for tf in testfiles:
            expected = {}
            expected = get_expected(tf)

            args["file_name"] = tf
            program = PrologFile(args['file_name'], parser=DCParser())
            solver = InferenceSolver(abe, **args)
            probabilities = solver.probability(program, **args)
            computed = {}
            for k,v in probabilities.items():
                computed[str(k)] = str(v.value)

            for query in expected :
                self.assertEqual(expected[query], computed[query], msg=query)



    def test_problog_system(self):
        path2files = root_path("test")
        testfiles = [os.path.join(path2files,f) for f in os.listdir(path2files) if os.path.isfile(os.path.join(path2files, f))]
        print(len(testfiles))

        long = [
            "12_holidays.pl",
            "6_hmm_weather.pl"
        ]
        fail = [
            "00_trivial_duplicate.pl",
            "findall_duplicates.pl",
            "smokers_person.csv",
            "extern_lib.py",
            "00_trivial_undefined.pl",
            "negative_cycle2.pl",
            "negative_query.pl",
            "11_ads_numerical.pl",
            "findall_check.pl",
            "consult_nested.pl",
            "smokers_csv.pl",
            "some_cycles.pl",
            "advars_smokers.pl",
            "bug_nonground_error.pl",
            "00_trivial_not.pl",
            "list_sample_multi.pl",
            "findall4.pl",
            "list_sample.pl",
            "4_bayesian_net.pl",
            "findall2.pl",
            "swap.pl",
            "call_notfound.pl",
            "negation.pl",
            "smokers.sqlite",
            "01_inconsistent.pl",
            "smokers_friend_of.csv"
        ]



        abe = "pyro"
        args = {"device":"cpu", "ttype":"float64", "n_samples":100}

        count = 0
        for tf in testfiles:

            count +=1
            print(tf, count)

            if tf.split("/")[-1] in fail:
                print(tf, count, "FAIL")
                continue
            elif tf.split("/")[-1] in long:
                print(tf, count, "LONG")
                continue
            else:
                print(tf, count)

            expected = get_expected(tf)
            args["file_name"] = tf


            try:
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
                self.assertEqual(expected, type(e).__name__)
            else :
                self.assertIsInstance( expected, dict )
                self.assertCountEqual(expected, computed)

                for query in expected:
                    self.assertAlmostEqual(float(expected[query]), computed[query], msg=query)

        print("failed tests: {}".format(len(fail)))


if __name__ == "__main__":
    unittest.main()
