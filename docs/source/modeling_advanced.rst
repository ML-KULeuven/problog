Writing models in ProbLog: advanced concepts
============================================

Debugging your ProbLog program
++++++++++++++++++++++++++++++

ProbLog does not support I/O as Prolog does.
It is however possible to write debugging information to the console using the ``debugprint`` builtin.
This builtin takes up to 10 arguments that will be printed to the console when the call is encountered during grounding.
The arguments are separated by spaces.


Calling Python from ProbLog
+++++++++++++++++++++++++++

ProbLog allows calling functions written in Python from a ProbLog model.
The functionality is provided by the ``problog.extern`` module.
This module introduces two decorators.

  * ``problog_export``: for deterministic functions (i.e. that return exactly one result)
  * ``problog_export_nondet``: for non-deterministic functions (i.e. that return any number of results)

These decorators take as arguments the types of the arguments.
The possible argument types are

  * ``str``: a string
  * ``int``: an integer number
  * ``float``: a floating point number
  * ``list``: a list of terms
  * ``term``: an arbitrary Prolog term

Each argument is prepended with a ``+`` or a ``-`` to indicate whether it is an input or an output argument.
The arguments should be in order with input arguments first, followed by output arguments.

The function decorated with these decorators should have exactly the number of input arguments and it should return a tuple
of length the number of output arguments.
If there is only one output argument, it should not be wrapped in a tuple.

Functions decorated with ``problog_export_nondet`` should return a list of result tuples.

For example, consider the following Python module ``numbers.py`` which defines two functions.

.. code-block:: python

    from problog.extern import problog_export, problog_export_nondet

    @problog_export('+int', '+int', '-int')
    def sum(a, b):
        """Computes the sum of two numbers."""
        return a + b

    @problog_export('+int', '+int', '-int', '-int')
    def sum_and_product(a, b):
        """Computes the sum and product of two numbers."""
        return a + b, a * b

    @problog_export_nondet('+int', '+int', '-int')
    def in_range(a, b):
        """Returns all numbers between a and b (not including b)."""
        return list(range(a, b))    # list can be empty

    @problog_export_nondet('+int')
    def is_positive(a):
        """Checks whether the number is positive."""
        if a > 0:
            return [()] # one result (empty tuple)
        else:
            return []   # no results


This module can be used in ProbLog by loading it using the ``use_module`` directive.

.. code-block:: prolog

    :- use_module('numbers.py').

    query(sum(2,4,X)).
    query(sum_and_product(2,3,X,Y)).
    query(in_range(1,4,X)).
    query(is_positive(3)).
    query(is_positive(-3)).

The result of this model is

.. code-block:: prolog

    in_range(1,4,1):        1
    in_range(1,4,2):        1
    in_range(1,4,3):        1
         sum(2,4,6):        1
       sum(2,3,5,6):        1
    is_positive(-3):        0
     is_positive(3):        1

It is possible to store persistent information in the internal database.
This database can be accessed as ``problog_export.database``.


Using continuous distributions (sampling only)
++++++++++++++++++++++++++++++++++++++++++++++

