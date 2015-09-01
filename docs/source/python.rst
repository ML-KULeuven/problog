Using ProbLog from Python
=========================

.. code-block:: python

    from problog.program import PrologFile
    from problog.formula import LogicFormula
    from problog.sdd_formula import SDD
    from problog.nnf_formula import NNF
    from problog.cnf_formula import CNF


    def problog_v1(model) :
        program = PrologFile(model)
        formula = LogicFormula.create_from(program)
        cnf = CNF.create_from(formula)
        nnf = NNF.create_from(cnf)
        return nnf.evaluate()


    def problog_v2(model) :
        program = PrologFile(model)
        formula = LogicFormula.create_from(program)
        sdd = SDD.create_from(formula)
        return sdd.evaluate()

