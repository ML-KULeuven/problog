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


Decision-Theoretic ProbLog
--------------------------

.. code-block:: python

    from problog.tasks.dtproblog import dtproblog
    from problog.program import PrologString

    model = """
        0.3::rain.
        0.5::wind.
        ?::umbrella.
        ?::raincoat.

        broken_umbrella :- umbrella, rain, wind.
        dry :- rain, raincoat.
        dry :- rain, umbrella, not broken_umbrella.
        dry :- not(rain).

        utility(broken_umbrella, -40).
        utility(raincoat, -20).
        utility(umbrella, -2).
        utility(dry, 60).
    """

    program = PrologString(model)
    decisions, score, statistics = dtproblog(program)

    for name, value in decisions.items():
        print ('%s: %s' % (name, value))
