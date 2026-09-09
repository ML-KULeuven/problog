"""
Part of the ProbLog distribution.

Copyright 2022 KU Leuven, DTAI Research Group

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
import os
import re
import subprocess
import tempfile
import unittest

from problog import root_path, system_info


class TestdSharpCompilation(unittest.TestCase):

    def test_dsharp_long_clause(self):
        """
        Bug description: https://github.com/QuMuLab/dsharp/issues/15
        Test dSharp on a CNF with a very long clause (>=66k chars)
        Model count should be exactly 1; and all variables (20058) should be used.
        Previously it erroneously printed only 20056.
        """
        if system_info.get("c2d", False):
            return  # skip if system uses c2d instead of dsharp
        # run dSharp
        filename = root_path("test/specific/", "long_clause.cnf")
        cmd = ["dsharp", filename]
        mc_output = subprocess.run(cmd, stdout=subprocess.PIPE).stdout.decode('utf-8')
        # extract statistics
        used_vars = re.search("used Variables:\t*([0-9]*)\n", mc_output)
        if used_vars is not None:
            expected_used_variables = "20058"
            self.assertEqual(expected_used_variables, used_vars.group(1))

    def test_dsharp_decomposability(self):
        """
        Bug description: https://github.com/ML-KULeuven/problog/issues/113

        dSharp could emit a d-DNNF that is logically equivalent to the input
        but not decomposable: an AND node with children sharing a variable.
        The search's own model count stays correct, so nothing notices, but
        every weighted model count taken off the .nnf is wrong -- ProbLog
        reported a probability of 156.13 for a model whose answer is 0.6.

        Decomposability is the property the whole .nnf format exists to
        provide, so check it directly on the binary we are about to use, and
        check that the .nnf counts to the same number the solver reports.
        """
        if system_info.get("c2d", False):
            return  # skip if system uses c2d instead of dsharp

        cnf = root_path("test/specific/", "dsharp_decomposability.cnf")
        fd, nnf = tempfile.mkstemp(".nnf")
        os.close(fd)
        try:
            out = subprocess.run(
                ["dsharp", "-Fnnf", nnf, cnf], stdout=subprocess.PIPE
            ).stdout.decode("utf-8")
            reported = re.search(r"# of solutions:\s*(\S+)", out)
            self.assertIsNotNone(reported, "dsharp did not report a solution count")
            self.assertAlmostEqual(798.0, float(reported.group(1)), places=3)

            non_decomposable, models = _nnf_stats(nnf)
            self.assertEqual(
                0,
                non_decomposable,
                "dsharp emitted %d non-decomposable AND node(s); the .nnf is not a "
                "d-DNNF and any weighted model count taken from it will be wrong"
                % non_decomposable,
            )
            self.assertEqual(
                798,
                models,
                "the .nnf has %d models but dsharp reports 798 solutions" % models,
            )
        finally:
            try:
                os.remove(nnf)
            except OSError:
                pass


def _nnf_stats(path):
    """Count non-decomposable AND nodes and models of a .nnf (c2d format).

    ORs are smoothed on the fly, so the count is over all variables.
    """
    nodes = []
    with open(path) as f:
        nvars = int(f.readline().split()[3])
        for line in f:
            p = line.split()
            if not p:
                continue
            if p[0] == "L":
                nodes.append(("L", int(p[1])))
            elif p[0] == "A":
                nodes.append(("A", [int(x) for x in p[2:]]))
            elif p[0] == "O":
                nodes.append(("O", [int(x) for x in p[3:]]))

    varset = [0] * len(nodes)
    count = [0] * len(nodes)
    non_decomposable = 0
    for i, nd in enumerate(nodes):
        if nd[0] == "L":
            varset[i], count[i] = 1 << abs(nd[1]), 1
        elif nd[0] == "A":
            seen, c = 0, 1
            for j in nd[1]:
                if varset[j] & seen:  # children must have disjoint variables
                    non_decomposable += 1
                seen |= varset[j]
                c *= count[j]
            varset[i], count[i] = seen, c
        else:
            seen = 0
            for j in nd[1]:
                seen |= varset[j]
            varset[i] = seen
            count[i] = sum(
                count[j] << bin(seen & ~varset[j]).count("1") for j in nd[1]
            )
    return non_decomposable, count[-1] << (nvars - bin(varset[-1]).count("1"))
