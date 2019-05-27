Using ProbLog from Python
=========================

The ``problog`` package can be used to interact with ProbLog directly from Python.

.. code-block:: python

    from problog.program import PrologString
    from problog import get_evaluatable

    model = """0.3::a.  query(a)."""
    result = get_evaluatable().create_from(PrologString(model)).evaluate()

The function ``problog.get_evaluatable()`` automatically selects a suitable knowledge compilation
representation.

The result is a dictionary which maps a query term on its probability.
In this case, the result is ``{a: 0.3}``.

This process can also be split up in different stages.

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

Decision-Theoretic ProbLog is implemented on top of the ProbLog core.
Here is an example on how to interact with it.

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


Parameter learning (LFI)
------------------------

.. code-block:: python

    from problog.logic import Term
    from problog.program import PrologString
    from problog.learning import lfi

    model = """
    t(0.5)::burglary.
    0.2::earthquake.
    t(_)::p_alarm1.
    t(_)::p_alarm2.
    t(_)::p_alarm3.

    alarm :- burglary, earthquake, p_alarm1.
    alarm :- burglary, \+earthquake, p_alarm2.
    alarm :- \+burglary, earthquake, p_alarm3.
    """

    alarm = Term('alarm')
    burglary = Term('burglary')
    earthquake = Term('earthquake')

    examples = [
        [(burglary, False), (alarm, False)],
        [(earthquake, False), (alarm, True), (burglary, True)],
        [(burglary, False)]
    ]

    score, weights, atoms, iteration, lfi_problem = lfi.run_lfi(PrologString(model), examples)

    print (lfi_problem.get_model())


Sampling
--------

Sampling is implemented on top of the ProbLog core.
Here is an example on how to interact with it.

.. code-block:: python

    from problog.tasks import sample
    from problog.program import PrologString

    modeltext = """
        0.3::a.
        0.5::b.
        c :- a; b.
        query(a).
        query(b).
        query(c).
    """

    model = PrologString(modeltext)
    result = sample.sample(model, n=3, format='dict')

The result is a list of dictionaries mapping the query atoms onto their sampled value.
In this case the result could be
``[{a: False, b: True, c: True}, {a: False, b: False, c: False}, {a: True, b: True, c: True}]``.

Sampling also supports continuous distributions.

.. code-block:: python

    from problog.tasks import sample
    from problog.program import PrologString

    modeltext = """
        uniform(0,10)::a.
        0.5::b.
        c :- value(a, A), A >= 3; b.
        query(a).
        query(b).
        query(c).
    """

    model = PrologString(modeltext)
    result = sample.sample(model, n=3, format='dict')

In this case the result could be
``[{a: 3.17654015834, b: False, c: True}, {a: 2.06136530868, b: False, c: False}, {a: 6.56599142521, b: False, c: True}]``

You can also add your own distributions.

.. code-block:: python

    from problog.tasks import sample
    from problog.program import PrologString

    modeltext = """
        my_uniform(0,10)::a.
        0.5::b.
        c :- value(a, A), A >= 3; b.
        query(a).
        query(b).
        query(c).
    """

    import random
    import math

    # Define a function that generates a sample.
    def integer_uniform(a, b):
        return math.floor(random.uniform(a, b))

    model = PrologString(modeltext)
    # Pass the mapping between name and function using the distributions parameter.
    result = sample.sample(model, n=3, format='dict', distributions={'my_uniform': integer_uniform})

Example output: ``[{a: 0.0, b: True, c: True}, {a: 7.0, b: False, c: True}, {a: 0.0, b: False, c: False}]``

