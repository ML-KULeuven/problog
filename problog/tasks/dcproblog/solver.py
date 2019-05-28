import os
from collections import OrderedDict
from graphviz import Source
from hashlib import md5

from problog.formula import LogicFormula
from problog.cycles import break_cycles

from .engine import init_engine, init_model
from .formula import LogicFormulaHAL
from .sdd_formula import SDDHAL
from .evaluator import SemiringHAL

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
    def __init__(self, abe_name, draw_diagram=False, dpath=None,  file_name=None, n_samples=None, ttype=None, device=None):
        self.operator = SumOperator()

        self.abstract_abe = AbstractABE(abe_name, n_samples=n_samples, ttype=ttype, device=device)
        self.draw_diagram = draw_diagram
        self.dpath = dpath
        self.file_name = file_name

    def get_density_queries(self, lf_hal):
        density_queries = lf_hal.density_queries
        _new_query_names = {}
        for k,v in lf_hal._names[lf_hal.LABEL_QUERY].items():
            if isinstance(k,tuple):
                _new_query_names[k] =v
            elif not k.functor=="density":
                _new_query_names[k] =v

        lf_hal._names[lf_hal.LABEL_QUERY] = _new_query_names
        return density_queries

    def ground(self, model, queries=None, **kwdargs):
        engine = init_engine(**kwdargs)
        lf_hal = LogicFormulaHAL(db=model)
        model, evidence, ev_target = init_model(engine, model, target=lf_hal)
        free_variables = lf_hal.free_variables
        lf_hal = LogicFormulaHAL.create_from(model, label_all=True, \
            propagate_evidence=True, engine=engine, queries=queries)
        density_queries = self.get_density_queries(lf_hal)
        density_values = lf_hal.density_values

        return lf_hal, density_queries, density_values, free_variables

    def compile_formula(self, lf,  **kwdargs):
        sdd_hal = SDDHAL(**kwdargs)
        diagram = sdd_hal.create_from(lf, label_all=True, **kwdargs)
        diagram.build_dd()
        return diagram


    def calculate_probabilities(self, sdds, semiring, dde, **kwdargs):
        probabilities = OrderedDict()
        e_evaluated = dde.evaluate_sdd(sdds["e"], semiring, normalization=True, evaluation_last=False)
        for q, qe_sdd in sdds["qe"].items():
            #if evalutation last true then sdd deref but produces error
            qe_evaluated = dde.evaluate_sdd(qe_sdd, semiring, evaluation_last=False)
            q_probability = semiring.algebra.probability(qe_evaluated, e_evaluated)
            probabilities[q] = q_probability
        return probabilities

    def make_diagram(self, dde, sdds):
        # evidence_inode = dde.evidence_inode
        # dot  = dde.formula.sdd_functions_to_dot(evidence_inode=dde.evidence_inode)
        dot  = dde.formula.sdd_functions_to_dot(sdds=sdds["qe"])
        g = Source(dot)
        if self.file_name:
            file_name = os.path.basename(os.path.normpath(self.file_name)).strip(".pl")
        else:
            file_name = graph
        if not self.dpath:
            filepath=os.getcwd()
        else:
            filepath = os.path.dirname(__file__)

        diagram_name = os.path.join(filepath,'diagrams/{}.gv').format(file_name)
        g.render(diagram_name, view=False)


    # from .utils import profile
    # @profile("tottime")
    def probability(self, program, **kwdargs):
        lf_hal, density_queries, density_values, free_variables = self.ground(program, queries=None, **kwdargs)
        lf = break_cycles(lf_hal, LogicFormulaHAL(**kwdargs))

        semiring = SemiringHAL(self.operator.get_neutral(), self.abstract_abe, density_values, density_queries, free_variables)
        diagram = self.compile_formula(lf, **kwdargs)


        dde = diagram.get_evaluator(semiring=semiring, **kwdargs)
        dde.formula.density_values = density_values
        sdds = dde.get_sdds()
        if self.draw_diagram:
            assert not dde.evidence_inode==None
            self.make_diagram(dde, sdds)

        probabilities = self.calculate_probabilities(sdds, semiring, dde, **kwdargs)
        return probabilities

    def print_result(self, probabilities):
        for query in probabilities:
            q_str = str(query)
            print("{query: >20}: {probability}".format(query=q_str, probability=probabilities[query].value))
