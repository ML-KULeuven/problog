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

test_base = root_path("problog", "tasks", "dcproblog", "test")


def get_expected(filename):
    results = {}
    with open(filename) as f:
        reading = False
        for l in f:
            l = l.strip()
            if l.startswith("%Expected outcome:"):
                reading = True
            elif reading:
                if l.lower().startswith("% error"):
                    return l[len("% error") :].strip()
                elif l.startswith("% "):
                    query, prob = l[2:].rsplit(None, 1)
                    results[query.strip()] = prob.strip()
                else:
                    reading = False
            if l.startswith("query(") and l.find("% outcome:") >= 0:
                pos = l.find("% outcome:")
                query = l[6:pos].strip().rstrip(".").rstrip()[:-1]
                prob = l[pos + 10 :]
                results[query.strip()] = float(prob.strip())

    return results


def get_filenames(*args):
    path2files = os.path.join(test_base, *args)
    testfiles = [
        os.path.join(path2files, f)
        for f in os.listdir(path2files)
        if os.path.isfile(os.path.join(path2files, f))
    ]
    return testfiles


class TestTasks(unittest.TestCase):
    def test_pyro_examples(self):
        testfiles = get_filenames("pyro", "examples")
        abe = "pyro"
        args = {"device": "cpu", "ttype": "float64", "n_samples": 50000}
        for tf in testfiles:
            expected = {}
            expected = get_expected(tf)

            args["file_name"] = tf
            program = PrologFile(args["file_name"], parser=DCParser())
            solver = InferenceSolver(abe, **args)

            probability = {}
            results = solver.probability(program, **args)
            probabilities = results["q"]

            computed = {}
            for k, v in probabilities.items():

                computed[str(k)] = float(v.value)

            for query in expected:
                self.assertAlmostEqual(
                    float(expected[query]), computed[query], places=2, msg=query
                )

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
            program = PrologFile(args["file_name"], parser=DCParser())
            solver = InferenceSolver(abe, **args)
            results = solver.probability(program, **args)
            probabilities = results["q"]
            computed = {}
            for k, v in probabilities.items():
                computed[str(k)] = str(v.value)

            for query in expected:
                self.assertEqual(expected[query], computed[query], msg=query)

    def test_pyro_problog_system(self):
        path2files = root_path("test")
        testfiles = [
            os.path.join(path2files, f)
            for f in os.listdir(path2files)
            if os.path.isfile(os.path.join(path2files, f)) and f.endswith(".pl")
        ]

        abe = "pyro"
        args = {"device": "cpu", "ttype": "float64", "n_samples": 100}
        NOTSUPPORTED = [
            "subquery.pl",
        ]
        for tf in testfiles:
            # print(tf)
            if tf.split("/")[-1] in NOTSUPPORTED:
                continue

            expected = get_expected(tf)
            args["file_name"] = tf
            try:
                program = PrologFile(args["file_name"], parser=DCParser())
                solver = InferenceSolver(abe, **args)
                results = solver.probability(program, **args)
                probabilities = results["q"]
                computed = {}
                for k, v in probabilities.items():
                    computed[str(k)] = float(v.value)
            except Exception as err:
                # print("exception %s" % err)
                e = err
                computed = None

            if computed is None:
                self.assertEqual(expected, type(e).__name__)
            else:
                self.assertIsInstance(expected, dict)
                self.assertCountEqual(expected, computed)

                for query in expected:
                    self.assertAlmostEqual(
                        float(expected[query]), computed[query], msg=query
                    )

    def test_psi_problog_system(self):
        try:
            import psipy
        except ImportError:
            return
        path2files = root_path("test")
        testfiles = [
            os.path.join(path2files, f)
            for f in os.listdir(path2files)
            if os.path.isfile(os.path.join(path2files, f)) and f.endswith(".pl")
        ]

        NOTSUPPORTED = [
            "subquery.pl",
        ]

        abe = "psi"
        args = {}
        for tf in testfiles:
            if tf.split("/")[-1] in NOTSUPPORTED:
                continue

            expected = get_expected(tf)
            args["file_name"] = tf
            try:
                program = PrologFile(args["file_name"], parser=DCParser())
                solver = InferenceSolver(abe, **args)
                results = solver.probability(program, **args)
                probabilities = results["q"]
                computed = {}
                for k, v in probabilities.items():
                    v = str(v.value)

                    if "/" in v:
                        v = float(v.split("/")[0]) / float(v.split("/")[1])
                    else:
                        v = float(v)
                    computed[str(k)] = v
            except Exception as err:
                # print("exception %s" % err)
                e = err
                computed = None

            if computed is None:
                self.assertEqual(expected, type(e).__name__)
            else:
                self.assertIsInstance(expected, dict)
                self.assertCountEqual(expected, computed)

                for query in expected:
                    self.assertAlmostEqual(
                        float(expected[query]), computed[query], msg=query
                    )


if __name__ == "__main__":
    unittest.main()
