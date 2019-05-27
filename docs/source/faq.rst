Frequently asked questions
==========================


I get a NonGroundProbabilisticClause error. What is going on?
-------------------------------------------------------------

This error occurs when there is a successful evaluation of the body of a probabilistic clause in which not all variables have been assigned.

ProbLog requires that, after grounding, all probabilistic facts are ground (i.e. contain no variables).
A simple example is

.. code-block:: prolog

    0.4 :: a(_).
    b :- a(_).
    query(b).

The argument of *a/1* is never instantiated.

Usually, this error will occur due to an automatic translation made by ProbLog when the model contains clauses with a probability.
An example is

.. code-block:: prolog

    0.3::a(_,_).
    0.4::c(X) :- a(X, Y).
    query(c(1)).

This is translated internally to the program

.. code-block:: prolog

    a(_,_).
    0.4::choice(0, 0, c(X),X,Y).
    c(X) :- a(X, Y), choice(0, 0, c(X),X,Y).
    query(c(1)).

For each probabilistic clause, a new probabilistic fact is created which contains all variables used in the clause (with a few exceptions, see below).
In this case, this includes the variable `Y` which is never instantiated, an thus remains non-ground.

.. rubric:: How to resolve?

There are two possible solutions:

 - make sure that the variable `Y` is instantiated during grounding,
 - remove variable `Y` from the clause by making a auxiliary predicate

.. warning::
    Note that these solutions are not equivalent!

The first solution can be achieved by explicitly listing the possible values of the second argument of `a/2`.

.. code-block:: prolog

    a(_,Y) :- Y = a; Y = b; Y = c.
    0.4::c(X) :- a(X, Y).
    query(c(1)).

Grounding this program will create three independent probabilistic facts, one for each value of `Y`.
The query is true if any of these facts is true.
The result of this program is therefore (1 - (1 - 0.4)^3) = **0.784**.

The second solution defines an auxiliary predicate which *hides* the variable `Y`.

.. code-block:: prolog

    a(_,_).
    exists_a(X) :- a(X, Y).
    0.4::c(X) :- exists_a(X).
    query(c(1)).

Grounding this program creates only one probabilistic fact.
The result is therefore **0.4**.

.. rubric:: Special cases

As mentioned above, each probabilistic clause is rewritten to a regular clause and a probabilistic fact.
The probabilistic fact contains all variables present in clauses except for variables that,
due to the context in which they occur, will never be instantiated.

Currently, there are two such cases:

  - variables uses in the pattern or goal (i.e. the first two arguments) of meta-predicates such as ``findall/3`` and ``all/3``.
  - variables only occuring within a negation

By Prolog's semantics such variables are not instantiated.


Here are some examples:

.. code-block:: prolog

    a(1,2).
    a(1,3).
    a(1,4).

    0.3 :: q(X, L) :- all(Y, a(X, Y), L).

    query(q(1,L)).

Result: q(1,[2,3,4]) with probability 0.3.

.. code-block:: prolog

    a(1,2).
    a(1,3).
    a(1,4).

    my_all(A,B,C) :- all(A,B,C).

    0.3 :: q(X, L) :- my_all(Y, a(X, Y), L).

    query(q(1,L)).

Result: NonGroundProbabilisticClause: the exception only holds for the builtins ``findall/3`` and ``all/3`` and not for the custom ``my_all/3``.

.. code-block:: prolog

    0.1::a(1,2).
    0.1::a(1,3).
    0.1::a(1,4).

    0.3 :: q(X) :- \+ a(X, Y).

    query(q(1)).

Result: q(1) with probability 0.2187: the variable `Y` only occurs in a negated context and is therefore ignored


.. code-block:: prolog

    0.1::a(1,2).
    0.1::a(1,3).
    0.1::a(1,4).

    0.3 :: q(X) :- X \== Y, \+ a(X, Y).

    query(q(1)).

Result: NonGroundProbabilisticClause: the variable `Y` also occurs in a normal context, but is not instantiated

.. code-block:: prolog

    0.1::a(1,2).
    0.1::a(1,3).
    0.1::a(1,4).

    0.3 :: q(X) :- X \= Y, \+ a(X, Y).

    query(q(1)).

Result: q(1) with probability 0.0: the body always fails so there is no successful evaluation