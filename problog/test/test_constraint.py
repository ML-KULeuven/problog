import glob
import sys
import unittest
from copy import deepcopy

from problog import root_path
from problog import tasks


def read_result(filename):
    results = []
    with open(filename) as f:
        result = {}
        reading = False
        for l in f:
            l = l.strip()
            if l.startswith("%Expected outcome:"):
                reading = True
            elif reading:
                if l.lower().startswith("% error"):
                    return l[len("% error") :].strip()
                elif l.startswith("% "):
                    try:
                        query, prob = l[2:].rsplit(None, 1)
                        result[query.strip()] = float(prob.strip())
                    except:
                        results.append(deepcopy(result))
                        result = {}
                else:
                    reading = False
            if l.startswith("query(") and l.find("% outcome:") >= 0:
                pos = l.find("% outcome:")
                query = l[6:pos].strip().rstrip(".").rstrip()[:-1]
                prob = l[pos + 10 :]
                result[query.strip()] = float(prob.strip())
        results.append(result)
    return results


class TestConstraints(unittest.TestCase):
    def test_constraints(self):
        for filename in glob.glob(root_path("test/constraints", "*.pl")):
            try:
                solutions = tasks.load_task("constraint").run(filename)
            except ImportError:
                sys.stderr.write(
                    "No flatzinc support - The constraint tests are not performed.\n"
                )
                return True

            solutions = [
                {str(k): v for k, v in solution if v > 0.0} for solution in solutions
            ]
            expected = read_result(filename)
            self.assertTrue(solutions == expected)
