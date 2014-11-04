Logic
*****

The logic package contains basic logic constructs.

The package directly offers access to:

* Term (see :class:`.Term`)
* Var (see :class:`.Var`)
* Constant (see :class:`.Constant`)
* unify (see :func:`.unify` )


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

.. autoclass:: problog.logic.basic.And
  :members:
  :undoc-members:
  
.. autoclass:: problog.logic.basic.Or
  :members:
  :undoc-members:
  
.. autoclass:: problog.logic.basic.Clause
  :members:
  :undoc-members:

.. autoclass:: problog.logic.basic.Not
  :members:
  :undoc-members:

Logic program
-------------

.. autoclass:: problog.logic.basic.LogicProgram
  :members: __iter__, __iadd__, createFrom
  
Implementations
+++++++++++++++

.. autoclass:: problog.logic.program.SimpleProgram

.. autoclass:: problog.logic.program.PrologFile

.. autoclass:: problog.logic.program.ClauseDB


Unification
-----------

.. automodule:: problog.logic.unification
  :members:
  :undoc-members:

Logic Formula
-------------

.. .. autoclass:: problog.logic.engine.LogicFormula
..     :members:
