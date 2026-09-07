"""
problog.maxsat - Interface to MaxSAT solvers
--------------------------------------------

Interface to MaxSAT solvers.

..
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
import os
import shutil

from . import root_path
from .errors import InstallError, ProbLogError
from .util import mktempfile, subprocess_check_output, Timer


class UnsatisfiableError(ProbLogError):
    def __init__(self):
        ProbLogError.__init__(
            self, "No solution exists that satisfies the constraints."
        )


class MaxSATSolver(object):
    def __init__(self, command):
        self.command = command

    def unavailable_reason(self):
        """Explain why this solver cannot be used.

        :return: a description of what is missing, or None if the solver can run
        """
        if shutil.which(self.command[0]) is None:
            return "'%s' was not found on the search path" % self.command[0]
        return None

    def is_available(self):
        """Whether this solver is actually usable on this system."""
        return self.unavailable_reason() is None

    @property
    def extension(self):
        return "cnf"

    def prepare_input(self, formula, **kwargs):
        return formula.to_dimacs(weighted=int, **kwargs)

    def process_output(self, output):
        for line in output.split("\n"):
            if line.startswith("v "):
                return list(map(int, line.split()[1:-1]))
        raise UnsatisfiableError()

    def call_process(self, inputf):
        filename = mktempfile("." + self.extension)
        with open(filename, "w") as f:
            f.write(inputf)
        return subprocess_check_output(self.command + [filename])

    def evaluate(self, formula, **kwargs):
        with Timer("Transform input"):
            inputf = self.prepare_input(formula, **kwargs)
        with Timer("Solver call"):
            output = self.call_process(inputf)
        with Timer("Transform output"):
            result = self.process_output(output)
        return result


class MIPMaxSATSolver(MaxSATSolver):
    def __init__(self, command):
        MaxSATSolver.__init__(self, command)

    @property
    def extension(self):
        return "lp"

    def prepare_input(self, formula, **kwargs):
        return formula.to_lp(**kwargs)


class SCIPSolver(MIPMaxSATSolver):
    def __init__(self):
        MaxSATSolver.__init__(self, ["scip", "-f"])

    def process_output(self, output):
        facts = set()
        in_the_zone = False
        for line in output.split("\n"):
            line = line.strip()
            if line.startswith("objective value"):
                in_the_zone = True
            elif in_the_zone:
                if not line:
                    return list(facts)
                else:
                    facts.add(int(line.split()[0][1:]))
        raise UnsatisfiableError()


class Sat4jSolver(MaxSATSolver):
    def __init__(self):
        self.jar = root_path("problog", "bin", "java", "sat4j-maxsat.jar")
        MaxSATSolver.__init__(self, ["java", "-jar", self.jar])

    def unavailable_reason(self):
        if shutil.which("java") is None:
            return "java was not found on the search path"
        elif not os.path.exists(self.jar):
            return "%s is not present (it is not shipped with ProbLog)" % self.jar
        return None


def _create_solver(name):
    """Construct a solver by name, without checking whether it can be run."""
    if name == "scip":
        return SCIPSolver()
    elif name == "sat4j":
        return Sat4jSolver()
    else:
        return MaxSATSolver(["maxsatz"])


def get_solver(prefer=None):
    """Get a MaxSAT solver.

    :param prefer: name of the solver to use (default: maxsatz)
    :raise InstallError: if the requested solver is not usable on this system
    """
    solver = _create_solver(prefer)
    reason = solver.unavailable_reason()
    if reason is not None:
        available = get_available_solvers()
        if available:
            hint = " Available solvers: %s" % ", ".join(available)
        else:
            hint = " No MaxSAT solver is available"
        raise InstallError(
            "The MaxSAT solver '%s' can not be used: %s.%s"
            % (prefer or "maxsatz", reason, hint)
        )
    return solver


def get_known_solvers():
    """Get the names of all solvers ProbLog knows how to call."""
    return ["maxsatz", "scip", "sat4j"]


def get_available_solvers():
    """Get the names of the solvers that can actually be run on this system."""
    return [name for name in get_known_solvers() if _create_solver(name).is_available()]
