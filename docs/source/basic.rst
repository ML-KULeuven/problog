Basic usage
===========

.. code-block:: python

    def problog_v1(model) :
        program = PrologFile( model )
        formula = LogicFormula.createFrom( program )      # ground / specify engine?
        cnf = CNF.createFrom( formula )
        nnf = NNF.createFrom( cnf )
        return nnf.evaluate()
        
.. code-block:: python

    def problog_v2(model) :
        program = PrologFile( model )
        formula = LogicFormula.createFrom( program )      # ground / specify engine?
        sdd = SDD.createFrom( formula )
        return sdd.evaluate()
        
.. code-block:: python

    def problog_v3(model) :
        return PrologFile( model ).evaluate()