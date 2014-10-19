Logic
*****

The logic package contains basic logic constructs.

The package directly offers access to:

* Term (see :class:`problog.logic.basic.Term`)
* Var (see :class:`problog.logic.basic.Var`)
* Constant (see :class:`problog.logic.basic.Constant`)
* unify (see :func:`problog.logic.unification.unify` )


Basic logic: terms, functions, constants and variables
------------------------------------------------------

.. automodule:: problog.logic.basic

.. autoclass:: problog.logic.basic.Term
  :members:
  :undoc-members:

.. autoclass:: problog.logic.basic.Constant
  :members:
  :undoc-members:

.. autoclass:: problog.logic.basic.Var
  :members:
  :undoc-members:



Unification
-----------

.. automodule:: problog.logic.unification
  :members:
  :undoc-members:

Logic program
-------------

Syntactic sugar
+++++++++++++++

This module provides some syntactic sugar for constructing clauses.

First, we create some predicates and variables::

    from problog.logic import Var
    ancestor = Lit.create('anc')
    parent = Lit.create('par')
    X = Var('X')
    Y = Var('Y')
    Z = Var('Z')
    leo3 = Lit('leo3')
    al2 = Lit('al2')
    phil = Lit('phil')  

We can then write the following program::

    db = ClauseDB()
    db += ( ancestor(X,Y) << parent(X,Y) )
    db += ( ancestor(X,Y) << ( parent(X,Z) & ancestor(Z,Y) ) )
    db += ( parent( leo3, al2 ) )
    db += ( parent( al2, phil ) )

The following operators are available:

  =========== =========== ============
   Prolog      Python      English
  =========== =========== ============
   ``:-``          ``<<``      clause
   ``,``           ``&``       and
   ``;``           ``|``       or
   ``\+``          ``~``       not
  =========== =========== ============

.. warning::
    
    Due to Python's operator priorities, the body of the clause has to be between parentheses.

Basic constructs
++++++++++++++++

.. autoclass:: problog.logic.program.And
  :members:
  :undoc-members:

.. autoclass:: problog.logic.program.Or
  :members:
  :undoc-members:
  
.. autoclass:: problog.logic.program.Not
  :members:
  :undoc-members:

.. autoclass:: problog.logic.program.Lit
  :members:
  :undoc-members:

.. autoclass:: problog.logic.program.Clause
  :members:
  :undoc-members:

Clause database
+++++++++++++++

.. autoclass:: problog.logic.program.ClauseDB
  :members: 
  :undoc-members:
  
