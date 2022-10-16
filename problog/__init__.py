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
import os
import sys
import typing

sys.setrecursionlimit(10000)

# Set the PATH and PYTHON_PATH variables
from .setup import set_environment, gather_info

set_environment()

system_info = gather_info()


def root_path(*args):
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", *args))


library_paths = [root_path("problog", "library")]


# Load all submodules. This has two reasons:
#   - initializes all transformations (@transform)
#   - makes it possible to just import 'problog' and then use
#       something like problog.program.PrologFile
from . import cnf_formula
from . import core
from . import engine
from . import evaluator
from . import formula
from . import logic
from . import ddnnf_formula
from . import parser
from . import program
from . import sdd_formula
from . import sdd_formula_explicit
from . import util
from . import bdd_formula
from . import forward
from . import cycles
from . import kbest
from . import tasks
from . import debug

_evaluatables: typing.Dict[str, typing.Type[evaluator.Evaluatable]] = {
    "sdd": sdd_formula.SDD,
    "sddx": sdd_formula_explicit.SDDExplicit,
    "bdd": bdd_formula.BDD,
    "nnf": ddnnf_formula.DDNNF,
    "ddnnf": ddnnf_formula.DDNNF,
    "kbest": kbest.KBestFormula,
    "fsdd": forward.ForwardSDD,
    "fbdd": forward.ForwardBDD,
}

_semirings: typing.Dict[str, typing.Type[evaluator.Semiring]] = {
    "prob": evaluator.SemiringProbability,
    "logprob": evaluator.SemiringLogProbability,
    "symbolic": evaluator.SemiringSymbolic,
}


def get_semirings() -> typing.Collection[str]:
    """Get the set of registered semirings."""
    return _semirings.keys()


def register_semiring(name: str, cls: typing.Type[evaluator.Semiring]):
    """Register a semiring in the semiring registry."""
    assert (
        name not in _semirings
    ), "A semi-ring with the same name is already registered."
    _semirings[name] = cls


def get_semiring(name: typing.Optional[str] = None) -> typing.Type[evaluator.Semiring]:
    """Lookup a semiring."""
    if name is None:
        return _semirings["prob"]
    return _semirings.get(name, None)


def get_evaluatables() -> typing.Collection[str]:
    return _evaluatables.keys()


def get_evaluatable(
    name: typing.Optional[str] = None,
    semiring: typing.Optional[evaluator.Semiring] = None,
) -> typing.Type[evaluator.Evaluatable]:
    if name is None:
        if semiring is None or semiring.is_dsp():
            return evaluator.EvaluatableDSP
        else:
            return formula.LogicNNF
    else:
        return _evaluatables[name]
