"""
Part of the ProbLog distribution.

Copyright 2026 KU Leuven, DTAI Research Group

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import shutil
import subprocess
import unittest

from problog import root_path

#: How many times to run the solver on the same input.  The failure this
#: guards against comes from whatever the stack happens to hold, so it shows
#: up in a few runs out of a hundred rather than every time.
RUNS = 100


class TestMaxsatz(unittest.TestCase):
    def test_header_parsed_the_same_every_run(self):
        """maxsatz gave different answers for one unchanged input file.

        build_simple_sat_instance() copies the 'p wcnf ...' header into an
        uninitialised stack buffer and never terminates it, so the sscanf()
        that reads it carries on past the header into whatever follows on the
        stack.  When that starts with a digit the %lli takes it, and the hard
        clause weight comes out an order of magnitude too large -- 28135 read
        as 281359.  Clauses the caller meant to be hard are then below the
        threshold, so the solver is free to violate them.

        That is not a crash and not obviously wrong output: the solver answers
        a question nobody asked and returns a model that breaks a hard clause.
        In ProbLog it surfaced as 'problog explain' summing an already counted
        solution twice and reporting a probability of 1.4 -- roughly one run in
        thirty, on some platforms only.

        maxsatz echoes the header it parsed, so check that, and check the
        answer it gives.  Both have to hold on every run.
        """
        if shutil.which("maxsatz") is None:
            self.skipTest("maxsatz is not available")

        cnf = root_path("test/specific/", "maxsatz_hard_weight.cnf")
        with open(cnf) as f:
            header = f.readline().strip()

        seen_info = set()
        seen_answer = set()
        for _ in range(RUNS):
            out = subprocess.run(
                ["maxsatz", cnf], stdout=subprocess.PIPE
            ).stdout.decode("utf-8")
            for line in out.splitlines():
                if line.startswith("c Instance info:"):
                    seen_info.add(line.split(":", 1)[1].strip())
                elif line.startswith("s "):
                    seen_answer.add(line.strip())

        self.assertEqual(
            {header},
            seen_info,
            "maxsatz read the header of %s as %s over %d runs; it says '%s'. "
            "A hard weight larger than the file's makes hard clauses soft."
            % (cnf, sorted(seen_info), RUNS, header),
        )
        # Both blocked cubes cover the formula, so nothing is left to satisfy.
        self.assertEqual(
            {"s UNSATISFIABLE"},
            seen_answer,
            "maxsatz answered %s over %d runs on one unchanged input; it has "
            "to answer the same thing every time" % (sorted(seen_answer), RUNS),
        )


if __name__ == "__main__":
    unittest.main()
