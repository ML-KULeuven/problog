Writing models in ProbLog: advanced concepts
============================================

Output and errors your ProbLog program
++++++++++++++++++++++++++++++++++++++

ProbLog does not support I/O as Prolog does.
It is however possible to write out some information using the following predicates:

   * ``debugprint/N``: takes up to 10 arguments which are printed followed by a new line
   * ``write/N``: takes up to 10 arguments which are printed out; removes quotes; no new line
   * ``nl/0``: writes a new line character
   * ``writenl/N``: same as ``write\N`` followed by ``nl``.
   * ``error/N``: raise a UserError based with a message composed of up to 10 arguments

Calling Python from ProbLog
+++++++++++++++++++++++++++

ProbLog allows calling functions written in Python from a ProbLog model.
The functionality is provided by the ``problog.extern`` module.
This module introduces two decorators.

  * ``problog_export``: for deterministic functions (i.e. that return exactly one result)
  * ``problog_export_nondet``: for non-deterministic functions (i.e. that return any number of results)
  * ``problog_export_raw``: for functions without clear distinction between input and output

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

Functions decorated with ``problog_export_raw`` should return a list of tuples where each tuple
contains a value for each argument listed in the specification.

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

ProbLog provides a library that offers a very simple interface to an SQLite database.

Assume we have an SQLite database ``friends.db`` with two tables:

    *person(name)*
        A list of persons.

    *friend_of(name1, name2, probability)*
        A list of friendship relations.

We can load this database into ProbLog using the library ``db`` and the predicate \
``sqlite_load(+Filename)``.

.. code-block:: prolog

    :- use_module(library(db)).
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


Using data from a CSV file
++++++++++++++++++++++++++

ProbLog provides a library that offers a simple interface to an CSV file.

Assume we have two CSV files ``person.csv`` and ``friend_of.csv`` \
containing data for two predicates:

    *person(name)*
        A list of persons.

    *friend_of(name1, name2, probability)*
        A list of friendship relations.

These file contain as columns the terms of the predicate and the first line \
are the column names.

.. code-block:: sh

    $ cat person.csv
    "name"
    "ann"
    "bob"
    $ cat friend_of.csv
    "p1","p2","prob"
    "ann","bob",0.2

We can load these files into ProbLog using the library ``db`` and the predicate \
``csv_load(+Filename, +Predicatename)``. 

.. code-block:: prolog

    :- use_module(library(db)).
    :- csv_load('person.csv', 'person').
    :- csv_load('friend_of.csv', 'friend_of').

This will create a two predicates, one for each file with as arity the number of columns.
We can thus write the following variation of the smokers examples:

.. code-block:: prolog

    :- use_module(library(db)).
    :- csv_load('person.csv', 'person').
    :- csv_load('friend_of.csv', 'friend_of').

    P :: influences(X, Y) :- friend_of(X, Y, P).

    0.3::smokes(X) :- person(X).       % stress
    smokes(X) :- influences(Y, X), smokes(Y).

The library will automatically translate a call to predicates ``person`` and ``friends_of`` into a query on the \
respective csv-file. For example, the call ``friend_of(ann, B, P)`` will be matched to all lines that match

.. code-block:: sh

    "ann",*,*


Using continuous distributions (sampling only)
++++++++++++++++++++++++++++++++++++++++++++++

When using the sampling mode from Python, you can add arbitrary distributions with specialized sampling algorithms.
This can be achieved by passing them to the sample function.

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
