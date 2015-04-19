Sampler for ProbLog 2.1
=======================


1. Concept
----------

This extension of ProbLog 2.1 allows the user to randomly sample from a distribution represented as a ProbLog program.
The sampling algorithm described here is query-based, that is, it generates possible values for all queries present in the program.

The goal of this work is to provide a *minimal effort extension of ProbLog* that is capable of sampling from programs with continuous distributions.


2. Approach
-----------

The algorithm operates during grounding using the standard ProbLog 2.1 grounding mechanism.
Given a query, it will search for all proofs for that query in a top-down fashion.
When an annotated (e.g. probabilistic) atom is encountered, it is replaced with an atom representing the value based on sampling a value based on the annotation.
The returned value is stored, such that any subsequent uses of the same atom will return the same value.


2.1 Simple probabilistic facts
++++++++++++++++++++++++++++++

When a probabilistic fact is encountered, it is replaced with ``true`` or ``fail`` according to its probability.

2.2 Continuous distributions
++++++++++++++++++++++++++++

When a clause or fact is annotated with a continuous distribution, the head atom is associated with a value.
This value is sampled randomly from the distribution associated with the fact.

In order to sample and use a value from a given annotated clause, we introduce the ``sample/2`` built-in.
This built-in takes a goal as its first element, and binds the sample value with its second argument.
The obtained value can then be used in the regular Prolog comparison tests.


3. Example
----------

The following example compares the length of a person with two thresholds.
If the person is smaller than the lower one, then they can see.
If the person is taller than the higher one, then they will hit their head.

The height of a person is assumed to be normally distributed with a different mean based on whether a person is male of female.

.. code-block:: prolog
    
    0.5::male(P); 0.5::female(P).
    
    normal(172,30)::height(P) :- male(P).
    normal(168,30)::height(P) :- female(P).
    
    hits_head(Person,Height) :- sample(height(Person),H), H >= Height.
    cant_see(Person,Height) :- sample(height(Person),H), H =< Height.
    
    query(height(p1)).
    query(hits_head(p1,190)).
    query(cant_see(p1,160)).


**Example output (10 samples):**

.. code-block:: prolog

    cant_see(p1,160).
    height(p1) = 148.668337357.

.. code-block:: prolog

    cant_see(p1,160).
    height(p1) = 150.458353301.

.. code-block:: prolog

    cant_see(p1,160).
    height(p1) = 152.10612219.

.. code-block:: prolog

    height(p1) = 170.876103857.

.. code-block:: prolog

    height(p1) = 173.848162049.
    
.. code-block:: prolog
    
    cant_see(p1,160).
    height(p1) = 105.264159763.

.. code-block:: prolog

    height(p1) = 195.157716623.
    hits_head(p1,190).
    
.. code-block:: prolog
    
    cant_see(p1,160).
    height(p1) = 129.142273266.

.. code-block:: prolog

    height(p1) = 163.853076917.

.. code-block:: prolog

    height(p1) = 174.950591707.


4. Supported distributions
--------------------------

Currently, the following distributions are supported:

* ``normal/2``
* ``poisson/1``
* ``exponential/1``
* ``beta/2``
* ``gamma/2``
* ``uniform/2`` (With range)
* ``constant/1`` (Always the given value)

Discrete distributions have been omitted because they can be simulated using annotated disjunctions.

5. Notes
--------

* Logic operations (and/or) on atoms with a continuous value are not allowed. These will raise an exception.
  This can occur when the bodies of clauses corresponding to the same annotated head are not mutually exclusive.
* Evidence is currently not supported.


6. Implementation
-----------------

The algorithm is implemented as an extension on top of ProbLog 2.1.
It does not require modification to the ProbLog core.
It contains the following components:

* a Python function ``sample_value`` that given a term (the annotation) returns a value.
* the implementation of the builtin ``sample/2``
* an implementation of LogicFormula that is responsible for maintaining the sampled values 

It operates completely during the grounding phase, which means that no knowledge compilation is performed.

The full implementation is available in the file ``sample.py``.




