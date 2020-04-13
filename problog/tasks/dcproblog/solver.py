import os
from collections import OrderedDict

from problog.formula import LogicFormula
from problog.errors import InconsistentEvidenceError

from .cycles import break_cycles
from .engine import init_engine
from .formula import LogicFormulaHAL
from .sdd_formula import SDDHAL
from .evaluator import SemiringHAL
from .logic import Mixture


class Operator(object):
    def __init__(self, neutral_element, name):
        self.neutral_element = neutral_element
        self.name = name

    def get_neutral(self):
        return self.neutral_element

    def __str__(self):
        return self.name


class SumOperator(Operator):
    def __init__(self):
        Operator.__init__(self, 0.0, "sum")


class AbstractABE(object):
    def __init__(self, name, n_samples=None, ttype=None, device=None):
        self.name = name
        self.n_samples = n_samples
        self.ttype = ttype
        self.device = device


class InferenceSolver(object):
    def __init__(
        self,
        abe_name,
        draw_diagram=False,
        dpath=None,
        file_name=None,
        n_samples=None,
        ttype=None,
        device=None,
    ):
        self.operator = SumOperator()

        self.abstract_abe = AbstractABE(
            abe_name, n_samples=n_samples, ttype=ttype, device=device
        )
        self.draw_diagram = draw_diagram
        self.dpath = dpath
        self.file_name = file_name

    def ground(self, model, queries=None, **kwdargs):
        engine = init_engine(**kwdargs)
        lf_hal = LogicFormulaHAL(db=model)
        model = engine.prepare(model)
        lf_hal = LogicFormulaHAL.create_from(
            model,
            label_all=True,
            propagate_evidence=True,
            engine=engine,
            queries=queries,
        )
        return lf_hal

    def compile_formula(self, lf, **kwdargs):
        sdd_hal = SDDHAL(**kwdargs)
        diagram = sdd_hal.create_from(lf, label_all=True, **kwdargs)
        diagram.build_dd()
        return diagram

    def calculate_probabilities(self, sdds, dde, **kwdargs):
        probabilities = OrderedDict()
        e_evaluated = dde.evaluate_sdd(
            sdds["e"], normalization=True, evaluation_last=False
        )
        if e_evaluated.value == dde.semiring.zero().value:
            raise InconsistentEvidenceError(context=": after evaluating evidence")
        probabilities["q"] = OrderedDict()
        for q, qe_sdd in sdds["qe"].items():
            # if evalutation last true then sdd deref but produces error
            qe_evaluated = dde.evaluate_sdd(qe_sdd, evaluation_last=False)
            q_probability = dde.semiring.algebra.probability(qe_evaluated, e_evaluated)
            probabilities["q"][q] = q_probability
        probabilities["dq"] = OrderedDict()
        for dq, dqe_sdds in sdds["dqe"].items():
            r = []
            for c in dqe_sdds.args:
                free_variable = c[0].name
                dqe_evaluated = dde.evaluate_sdd(
                    c[1], free_variable=free_variable, evaluation_last=False
                )
                dq_evaluated = dde.semiring.algebra.probability(
                    dqe_evaluated, e_evaluated
                )
                r.append(dq_evaluated)
            probabilities["dq"][dq] = Mixture(*r)
        return probabilities

    def make_diagram(self, dde, sdds):
        # evidence_inode = dde.evidence_inode
        # dot  = dde.formula.sdd_functions_to_dot(evidence_inode=dde.evidence_inode)
        dot = dde.formula.sdd_functions_to_dot(sdds=sdds["qe"])
        if self.file_name:
            file_name = os.path.basename(os.path.normpath(self.file_name)).strip(".pl")
        else:
            file_name = graph
        if not self.dpath:
            # filepath=os.getcwd()
            filepath = "."

        else:
            filepath = os.path.dirname(__file__)

        diagram_name = os.path.join(filepath, "diagrams/{}.gv").format(file_name)

        try:
            from graphviz import Source

            g = Source(dot)
            g.render(diagram_name, view=False)
        except:
            pass

    def probability(self, program, **kwdargs):
        lf_hal = self.ground(program, queries=None, **kwdargs)
        lf_hal = break_cycles(
            lf_hal,
            LogicFormulaHAL(
                density_values=lf_hal.density_values,
                density_names=lf_hal.density_names,
                **kwdargs
            ),
        )
        semiring = SemiringHAL(
            self.operator.get_neutral(), self.abstract_abe, lf_hal.density_values
        )
        diagram = self.compile_formula(lf_hal, **kwdargs)
        dde = diagram.get_evaluator(semiring=semiring, **kwdargs)

        sdds = dde.get_sdds()
        if self.draw_diagram:
            assert not dde.evidence_inode == None
            self.make_diagram(dde, sdds)

        probabilities = self.calculate_probabilities(sdds, dde, **kwdargs)
        return probabilities

    def print_result(self, probabilities):
        for query in probabilities["q"]:
            q_str = str(query)
            print(
                "{query: >20}: {probability}".format(
                    query=q_str, probability=probabilities["q"][query].value
                )
            )
        for dquery in probabilities["dq"]:
            q_str = str(dquery)
            print(
                "{query: >20}: {probability}".format(
                    query=q_str, probability=probabilities["dq"][dquery]
                )
            )
