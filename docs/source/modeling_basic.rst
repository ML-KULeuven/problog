ProbLog models
=========================================

Prolog
------

The ProbLog modeling language is based on Prolog.

For a very quick introduction to Prolog you can check the `Wikipedia page <https://en.wikipedia.org/wiki/Prolog>`_.

For a more in-depth introduction you can check the
`Learn Prolog Now! <http://lpn.swi-prolog.org/lpnpage.php?pagetype=html&pageid=lpn-htmlch1>`_
tutorial or the book `Simply Logical by Peter Flach <https://book.simply-logical.space>`_.

ProbLog
-------

ProbLog extends Prolog syntax by introducing few new operators to allow for probabilistic modeling.
The following table provides a simple overview.

========================  =============================
   Definition                            Example
========================  =============================
fact                           ``a.``
probabilistic fact                ``0.5::a.``
clause                          ``a :- x.``
probabilistic clause        ``0.5::a :- x.``
annotated disjunction        ``0.5::a; 0.5::b.``
annotated disjunction         ``0.5::a; 0.5::b :- x.``
========================  =============================




In addition, Problog introduces also two predicates ``query`` and ``evidence`` for querying a probabilistic program or to condition
it to some pieces of evidence.



Probabilistic facts
+++++++++++++++++++

The main difference between Prolog and ProbLog is that ProbLog support probabilistic
facts. In the language, this extension is realized by the addition of a single operator ``::``.

In an example program involving coin tosses, we could have the following statement.

.. code-block:: prolog

    0.5::heads.

This indicates that the fact `heads` is true with probability 0.5 and false with probability 1-0.5.

This statement introduces *one* probabilistic choice.
If we want to model two coins, we need two separate facts:

.. code-block:: prolog

    0.5::heads1.
    0.5::heads2.

We can generalize this to an unbound number of coins by using a variable argument:

.. code-block:: prolog

    0.5::heads(C).



Annotated disjunctions
++++++++++++++++++++++

ProbLog also supports non-binary choices.
For example, we can model the throw of die as follows.

.. code-block:: prolog

    1/6::die(D, 1); 1/6::die(D, 2); 1/6::die(D, 3);
    1/6::die(D, 4); 1/6::die(D, 5); 1/6::die(D, 6).

This type of statement is called an *annotated disjunction*.
It expresses that at most one of these choices is true.
There is always an implicit *null* choice which states that none of the options is taken.
In this example, however, that extra state has zero probability because the probabilities of the
other states sum to one.



Probabilistic clauses
+++++++++++++++++++++

ProbLog also supports probabilities in the head of clauses.

.. code-block:: prolog

    0.1::burglary.
    0.9::alarm :- burglary.

This means that if burglary is true, alarm will be true as well with 90% probability.
Such a program can always be transformed into a program with just probabilistic facts.

.. code-block:: prolog

    0.1::burglary.
    0.9::alarm_on_burglary.

    alarm :- burglary, alarm_on_burglary.

Similarly, annotated disjunctions can also be used as head of a clause.

.. code-block:: prolog

    0.5::weather(0,sun); 0.5::weather(0,rain).
    0.8::weather(T,sun); 0.2::weather(T,rain) :- T > 0, T1 is T - 1, weather(T1, sun).
    0.4::weather(T,sun); 0.6::weather(T,rain) :- T > 0, T1 is T - 1, weather(T1, rain).

This program can also be transformed into an equivalent program with only annotated
disjunctive facts.

.. code-block:: prolog

    0.5::weather(0,sun); 0.5::weather(0,rain).

    0.8::weather_after_sun(T,sun); 0.2::weather_after_sun(T,rain).
    weather(T, sun) :- T > 0, T1 is T - 1, weather(T1, sun), weather_after_sun(T, sun).
    weather(T, rain) :- T > 0, T1 is T - 1, weather(T1, sun), weather_after_sun(T, rain).

    0.4::weather_after_rain(T,sun); 0.6::weather_after_rain(T,rain).
    weather(T, sun) :- T > 0, T1 is T - 1, weather(T1, sun), weather_after_rain(T, sun).
    weather(T, rain) :- T > 0, T1 is T - 1, weather(T1, sun), weather_after_rain(T, rain).


Queries
+++++++
A query indicates for which entity we want to compute the probability.

Queries are specified by adding a fact ``query(Query)``:

.. code-block:: prolog

    0.5::heads(C).
    two_heads :- heads(c1), heads(c2).
    query(two_heads).

Queries can also be added in batch.

.. code-block:: prolog

    0.5::heads(C).
    query(heads(C)) :- between(1, 4, C).

This will add the queries ``heads(1)``, ``heads(2)``, ``heads(3)`` and ``heads(4)``.

It is also possible to give a non-ground query, on the condition that the program itself contains
sufficient information to ground the probabilistic parts.

.. code-block:: prolog

    0.5::heads(C) :- between(1, 4, C).
    query(heads(C)).

This has the same effect as the previous program.


Evidence
++++++++

Evidence specifies any observations on which we want to condition this probability.
Evidence conditions a part of the program to be true or false.

It can be specified using a fact ``evidence(Literal)``.

.. code-block:: prolog

    0.5::heads(C).
    two_heads :- heads(c1), heads(c2).
    evidence(\+ two_heads).
    query(heads(c1)).

This program computes the probability that the first coin toss produces heads when we know
that the coin tosses did not both produce heads.
You can try it out in the `online editor <https://dtai.cs.kuleuven.be/problog/editor.html#task=prob&hash=aeb6af5c90ea198a9f933516e5710fbe>`_.

Evidence can also be specified using the binary predicate ``evidence(Positive, true)`` and
``evidence(Positive, false)``.


Tabling
+++++++

In ProbLog everything is tabled (or memoized).
Tabling is an advanced form of caching that is used to speed-up the execution of logic programs and
that allows certain types of cyclic programs.

Consider for example the following program that computes Fibonacci numbers.

.. code-block:: prolog

    fib(1, 1).
    fib(2, 1).
    fib(N, F) :-
        N > 2,
        N1 is N - 1,
        N2 is N - 2,
        fib(N1, F1),
        fib(N2, F2),
        F is F1 + F2.

In standard Prolog the execution time of this program is exponential in the size of N because
computations are not reused between recursive calls.
In tabled Prolog, the results of each computation is stored and reused when possible.
In this way, the above program becomes linear.

The previous example shows the power of caching, but tabling goes further than that.
Consider the following program that defines the ancestor relation in a family tree.

.. code-block:: prolog

    parent(ann, bob).
    parent(ann, chris).
    parent(bob, derek).

    ancestor(X, Y) :- ancestor(X, Z), parent(Z, Y).
    ancestor(X, Y) :- parent(X, Y).

We want to find out the descendents of Ann (i.e. the query `ancestor(ann, X)`).
In standard Prolog this program goes into an infinite recursion because the call to
`ancestor(ann, X)` leads immediately back to the equivalent call `ancestor(ann, Z)`.

In tabled Prolog, the identical call is detected and postponed,
and the correct results are produced.

Another example is that of finding a path in a (possibly cyclic) graph.
In ProbLog (or any other tabled Prolog) you can simply write.

.. code-block:: prolog

    path(X, Y) :- edge(X, Y).
    path(X, Y) :- edge(X, Z), path(Z, Y).

Control predicates
++++++++++++++++++

ProbLog uses Prolog to generate a ground version of a probabilistic logic program.
However, it does not support certain features that have no meaning in a probabilistic setting.
This includes cuts (``!``) and any other mechanism that breaks the pure logic interpretation of the
program.

For a full list of features that ProbLog does (not) support, please check :doc:`this section <prolog>`.

Findall
+++++++

ProbLog supports the meta-predicate ``findall/3`` for collecting all results to a query.
It is similar to ``findall/3`` in Prolog, but it eliminates duplicate solutions
(so it corresponds to ``all/3`` in YAP Prolog).

Note that the use of findall can lead to a combinatorial explosion when used in a probabilistic
context.



Other modes syntax
------------------

When ProbLog is executed in modes that are different from standard inference, new specific notation is available.



Learning from interpretations (LFI) mode
++++++++++++++++++++++++++++++++++++++++

ProbLog programs can be used in a learning setting, where some or all the probabilities are unkonwn.
In this case, the probability annotation in a probabilistic fact can be one of three possible forms:

- Of the form ``t(_)``, as in for instance ``t(_)::p_alarm1``. This indicates that the probability of this fact is to be learned from data.
- Of the form ``t(p)``, with ``p`` a probability, as in for instance ``t(0.5)::burglary``. This again indicates that the probability of this fact is to be learned from data, but instead of initializing this probability randomly, it will be set to the value ``p`` in the first iteration of EM.
- Of the form ``p``, with ``p`` a probability, as in for instance ``0.2::earthquake``. This indicates that the probability of this fact is fixed (not learned), and it correspond to the standard annotation of probabilistic facts.


In a learning setting, the ProbLog model is usually accompanied with a set of examples to learn from.
Examples are provided using the ``evidence`` predicate for each atom in an example.
Examples are separated using dashes ``---``.


An example of learning model:

.. code-block:: prolog

    t(_)::heads1.
    t(_)::heads2.
    someHeads :- heads1.
    someHeads :- heads2.

An example of how to provide examples:

.. code-block:: prolog

    evidence(someHeads,false).
    evidence(heads1,false).
    ----------------
    evidence(someHeads,true).
    evidence(heads1,true).
    ----------------
    evidence(someHeads,true).
    evidence(heads1,false).
    ----------------




Decision-theoretic mode
+++++++++++++++++++++++


DTProbLog is a decision-theoretic extension of ProbLog.

A model in DTProbLog differs from standard ProbLog models in a number of ways:

  * There are no queries and evidence.
  * Certain facts are annotated as being a decision fact for which the optimal choice must be determined.
  * Certain atoms are annotated with an utility, indicating their contribution to the final score.

Decision facts can be annotated in any of the following ways:

.. code-block:: prolog

   ?::a.
   decision(a).

Utilities can be defined using the ``utility/2`` predicate:

.. code-block:: prolog

   utility(win, 10).
   utility(buy, -1).



Libraries and Builtins
----------------------

ProbLog has a number of builtins and libraries available that simplify modeling.
An overview can be found on the page :doc:`prolog`.
