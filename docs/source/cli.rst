Using ProbLog as a standalone tool
==================================

The command line interface (CLI) gives access to the basic functionality of ProbLog 2.1.
It is accessible through the script ``problog`` (or ``problog-cli.py`` in the repository version).

The CLI has different modes of operation. These can be accessed by adding a keyword as the first \
argument.

Currently, the following modes are supported

  * (default, no keyword): standard ProbLog inference
  * ``sample``: generate samples from a ProbLog program
  * ``mpe``: most probable explanation
  * ``lfi``: learning from interpretations
  * ``dt``: decision-theoretic problog
  * ``explain``: evaluate using mutually exclusive proofs
  * ``ground``: generate a ground program
  * ``bn``: export a Bayesian network
  * ``shell``: interactive shell
  * ``install``: run the installer
  * ``unittest``: run the testsuite
  * ``web``: start a web server

Default (no keyword)
--------------------

In the default mode, ProbLog takes a model and computes the probability of the queries.
The output is a set of probabilities.

For example, given a file ``some_heads.pl``

.. code-block:: prolog

    $ cat some_heads.pl
    0.5::heads1.
    0.6::heads2.
    someHeads :- heads1.
    someHeads :- heads2.

    query(someHeads).

We can do

.. code-block:: prolog

    $ problog some_heads.pl
    someHeads : 0.8

ProbLog supports supports several alternative knowledge compilation tools.
By default, it uses the first available option from

    1. SDD
    2. d-DNNF using c2d
    3. d-DNNF using dsharp

The choice between SDD and d-DNNF can also be set using the option ``-k`` or ``--knowledge``.

By default, ProbLog transforms probabilities to logarithmic space to avoid rounding errors. \
This behavior can be disabled using the flag ``--nologspace``.

By default, Problog outputs its results to standard output. To write to an output file, use the \
option ``-o`` or ``--output``.

A time out can be set using the option ``-t`` or ``--timeout``.

For progress information use the ``-v`` or ``--verbose`` option (can be repeated).

The following options are advanced options:

  * ``--debug``: turn on debugging mode (prints full error messages)
  * ``--recursion-limit <value>``: increase Python's recursion limit (default: 1000)
  * ``--engine-debug``: turn on very verbose grounding
  * ``--sdd-auto-gc``: turn on SDD minimization and garbage collection (default: off)
  * ``--sdd-preset-variables``: preserve SDD variables (default: off)


Sampling and sampling based inference (``sample``)
--------------------------------------------------

Sampling
++++++++

In the ``sample`` mode, ProbLog will generate possible assignments to the queries in the model.
For example,

.. code-block:: prolog

    $ problog sample some_heads.pl -N 3
    ====================
    % Probability: 0.2
    ====================
    someHeads.
    % Probability: 0.2
    ====================
    someHeads.
    % Probability: 0.3

The probability indicated is the probability of *the choices made to obtain the sample*.
It is **NOT** the probability of the sample itself (because there may be multiple choices that \
lead to the same sample).

The argument ``-N`` indicates the number of samples to generate.

The argument ``--oneline`` can be used to change the output format to place each sample on a \
separate line. The previous output would then be formatted as:

.. code-block:: prolog

    $ problog sample some_heads.pl -N 3 --oneline
    % Probability: 0.2
    someHeads. % Probability: 0.2
    someHeads. % Probability: 0.3

By default, only query atoms are part of the sample.
To also include facts that were chosen while sampling, the argument ``--with-facts`` can be used.
The result above would then become

.. code-block:: prolog

    $ problog sample some_heads.pl -N 3 --oneline --with-facts
    % Probability: 0.2
    heads1. someHeads. % Probability: 0.2
    heads2. someHeads. % Probability: 0.3

The sampling algorithm supports **evidence** through rejection sampling.  All generated samples \
are guaranteed to satisfy the evidence.  Note that this process can be slow if the evidence has \
low probability.

The sampling algorithm support evidence propagation, that is, in certain cases it can ensure the \
 evidence holds without the use of rejection sampling.
To enable this feature use the ``--propagate-evidence`` argument. Evidence propagation is not \
 supported on programs with continuous distributions, or on programs where the evidence has \
 infinite support.

Sample based inference
++++++++++++++++++++++

It is also possible to use the sample mode for *probability estimation* by setting the flag \
``--estimate``.  The output is similar to the output in default mode.

The number of samples used for estimation can be determined in three ways:

    * by supplying the number of samples using the argument ``-N``
    * by supplying a timeout using the argument ``--timeout`` or ``-t`` (not supported on Windows)
    * by manually interrupting the process using CTRL-C or by sending a TERM(15) signal

.. code-block:: prolog

    $ problog sample some_heads.pl  --estimate -t 5
    % Probability estimate after 7865 samples:
    someHeads : 0.79249841

This mode also support the ``--propagate-evidence`` flag.

References:
+++++++++++

    https://lirias.kuleuven.be/handle/123456789/510199


Most Probable Explanation (``mpe``)
-----------------------------------

In MPE mode, ProbLog computes the possible world with the highest probability in which all queries
and evidence is true.

For example, consider the following program.

.. code-block:: prolog

    0.6::edge(1,2).
    0.1::edge(1,3).
    0.4::edge(2,5).
    0.3::edge(2,6).
    0.3::edge(3,4).
    0.8::edge(4,5).
    0.2::edge(5,6).

    path(X,Y) :- edge(X,Y).
    path(X,Y) :- edge(X,Z),
                 Y \== Z,
                 path(Z,Y).

    evidence(path(1,5)).
    evidence(path(1,6)).

This program describes a probabilistic graph.

.. digraph:: probabilistic_graph

    rankdir=LR;
    1 -> 3 [label="0.1"];
    1 -> 2 [label="0.6"];
    2 -> 5 [label="0.4"];
    2 -> 6 [label="0.3"];
    3 -> 4 [label="0.3"];
    4 -> 5 [label="0.8"];
    5 -> 6 [label="0.2"];

The command ``problog mpe pgraph.pl`` produces the most probable graph in which there are paths
from node 1 to node 5 and from node 1 to node 6.

The result is

.. code-block:: prolog

    edge(4,5)
    edge(1,2)
    edge(2,5)
    edge(2,6)
    \+edge(1,3)
    \+edge(3,4)
    \+edge(5,6)
    % Probability: 0.0290304

Note that the first edge is not necessary for the paths to exist, but it is included because it is
more likely to exist.

.. code-block:: prolog

    \+edge(3,4)
    edge(4,5)
    \+edge(1,3)
    edge(1,2)
    edge(2,5)
    \+edge(5,6)
    edge(2,6)


In order to compute the result, ProbLog uses a Max-Sat solver (``maxsatz``) which is included in
the distribution.


Learning from interpretations (``lfi``)
---------------------------------------

Learning expects a program with a number of unknown probabilities expressed as ``t(_)``.
If you want to start learning from a given initialisation, say 0.2, you can use ``t(0.2)``.

Given a program ``some_heads.pl`` with unknown probabilities:

.. code-block:: prolog

    t(_)::heads1.
    t(_)::heads2.
    someHeads :- heads1.
    someHeads :- heads2.

And an evidence file ``some_heads_ev.pl`` (sampled using probabilities 0.5 and 0.6, \
no evidence on ``heads2``):

.. code-block:: prolog

    evidence(someHeads,false).
    evidence(heads1,false).
    ----------------
    evidence(someHeads,true).
    evidence(heads1,true).
    ----------------
    evidence(someHeads,true).
    evidence(heads1,true).
    ----------------
    evidence(someHeads,false).
    evidence(heads1,false).
    ----------------
    evidence(someHeads,true).
    evidence(heads1,true).
    ----------------
    evidence(someHeads,true).
    evidence(heads1,false).
    ----------------
    evidence(someHeads,true).
    evidence(heads1,false).
    ----------------
    evidence(someHeads,true).
    evidence(heads1,true).
    ----------------
    evidence(someHeads,true).
    evidence(heads1,false).
    ----------------
    evidence(someHeads,true).
    evidence(heads1,false).

We can now learn the missing probabilities using:

.. code-block:: shell

    $ problog lfi some_heads.pl some_heads_ev.pl -O some_heads_learned.pl
    -6.88403875238 [0.4, 0.66666619] [t(_)::heads1, t(_)::heads2] 14

The learned program is saved in ``some_heads_learned.pl``.

.. code-block:: shell

    $ cat some_heads_learned.pl
    0.4::heads1.
    0.666666192095::heads2.
    someHeads :- heads1.
    someHeads :- heads2.




Decision Theoretic ProbLog (``dt``)
-----------------------------------

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


The current implementation supports two evaluation strategies: exhaustive search (exact) and local search (approximate).
Exhaustive search is the default.
Local search can be enabled with the argument ``-s local``.


References:

    https://lirias.kuleuven.be/handle/123456789/270066



Explanation mode (``explain``)
------------------------------

The ``explain`` mode offers insight in how probabilities can be computed for a ProbLog program.
Given a model, the output consists of three parts:

  * a reformulation of the model in which annotated disjunctions and probabilistic clauses are rewritten
  * for each query, a list of mutually exclusive proofs with their probability
  * for each query, the success probability determined by taking the sum of the probabilities of the individual proofs

This mode currently does not support evidence.

Grounding (``ground``)
----------------------

The ``ground`` mode provides access to the ProbLog grounder.
Given a model, the output consists of the ground program.

The output can be formatted in different formats:

  * pl: ProbLog format
  * dot: GraphViz representation of the AND-OR tree
  * svg: GraphViz representation of the AND-OR tree as SVG (requires GraphViz)
  * cnf: DIMACS encoding as CNF
  * internal: Internal representation (for debugging)

These can be provided using the ``--format`` option.

By default, the output is the ground program before cycle breaking (except for ``cnf``).
To perform cycle breaking, provide the ``--break-cycles`` argument.


Interactive shell (``shell``)
-----------------------------

ProbLog also has an interactive shell, similar to Prolog.
You can start it using the keyword ``shell`` as first command line argument.

The shell allows you to load models and query them interactively.

To load a file:

.. code-block:: prolog

    ?- consult('test/3_tossing_coin.pl').

Queries can be specified as in Prolog:

.. code-block:: prolog

    ?- heads(X).
    X = c4,
    p: 0.6;
    ---------------
    X = c3,
    p: 0.6;
    ---------------
    X = c2,
    p: 0.6;
    ---------------
    X = c1,
    p: 0.6;
    ---------------

.. code-block:: prolog

    ?- someHeads.
    p: 0.9744;
    ---------------

Evidence can be specified using a pipe (``|``):

.. code-block:: prolog

    ?- someHeads | not heads(c1).

Type ``help.`` for more information.


Bayesian network (``bn``)
-------------------------

ProbLog can export a program to a Bayesian network for comparison and
verification purposes. The grounded program that is exported is defined by the
query statements present in the program. The resulting network is not guaranteed
to be the most efficient representation and includes additional latent variables
to be able to express concepts such as annotated disjunctions. Decision nodes
are not supported.

.. code-block:: prolog

    $ ./problog-cli.py bn some_heads.pl --format=xdsl -o some_heads.xdsl

The resulting file can be read by tools such as
`GeNIe and SMILE <https://dslpitt.org>`_,
`BayesiaLab <http://www.bayesialab.com>`_,
`Hugin <http://www.hugin.com>`_ or
`SamIam <http://reasoning.cs.ucla.edu/samiam/>`_
(depending on the chosen output format).


Installation (``install``)
--------------------------

Run the installer.  This installs the SDD library.
This currently only has effect on Mac OSX and Linux.


Web server (``web``)
--------------------

Starts the web server.

To load libraries locally (no internet connection required), use ``--local``.
To open a web-browser with the editor use ``--browser``.


Testing (``unittest``)
----------------------

Run the unittests.

