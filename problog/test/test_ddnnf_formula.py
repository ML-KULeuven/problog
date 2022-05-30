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
import re
import subprocess
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
