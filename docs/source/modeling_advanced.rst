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

Using data from an SQLite database
++++++++++++++++++++++++++++++++++

ProbLog offers a library that offers a very simple interface to an SQLite database.

Assume we have an SQLite database ``friends.db`` with two tables:

    *person(name)*
        A list of persons.

    *friend_of(name1, name2, probability)*
        A list of friendship relations.

We can load this database into ProbLog using the library ``sqlite`` and the predicate \
``sqlite_load(+Filename)``.

.. code-block:: prolog

    :- use_module(library(sqlite)).
    :- sqlite_load('friends.db').

This will create a predicate for each table in the database with as arity the number of columns \
of that table.
We can thus write the following variation of the smokers examples:

.. code-block:: prolog

    :- use_module(library(sqlite)).
    :- sqlite_load('friends.db').

    P :: influences(X, Y) :- friend_of(X, Y, P).

    0.3::smokes(X) :- person(X).       % stress
    smokes(X) :- influences(Y, X), smokes(Y).

The library will automatically translate a call to a database predicate into a query on the \
database, for example, the call ``friend_of(ann, B, P)`` will be translated to the query

.. code-block:: sql

    SELECT name1, name2, probability FROM friend_of WHERE name1 = 'ann'


Using continuous distributions (sampling only)
++++++++++++++++++++++++++++++++++++++++++++++

