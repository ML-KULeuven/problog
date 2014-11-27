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

.. automodule:: problog.logic

.. autoclass:: problog.logic.Term
  :members:
  :undoc-members:

.. autoclass:: problog.logic.Constant
  :members:
  :undoc-members:

.. autoclass:: problog.logic.Var
  :members:
  :undoc-members:

.. autoclass:: problog.logic.And
  :members:
  :undoc-members:
  
.. autoclass:: problog.logic.Or
  :members:
  :undoc-members:
  
.. autoclass:: problog.logic.Clause
  :members:
  :undoc-members:

.. autoclass:: problog.logic.Not
  :members:
  :undoc-members:

Logic program
-------------

.. autoclass:: problog.logic.LogicProgram
  :members: __iter__, __iadd__, createFrom
  
Implementations
+++++++++++++++

.. autoclass:: problog.program.SimpleProgram

.. autoclass:: problog.program.PrologFile

.. autoclass:: problog.program.ClauseDB


Logic Formula
-------------

.. autoclass:: problog.formula.LogicFormula
     :members:
