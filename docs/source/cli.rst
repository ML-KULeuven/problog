Using ProbLog as a standalone tool
==================================

The command line interface (CLI) gives access to the basic functionality of ProbLog 2.1.
It is accessible through the script ``problog-cli.py`` or ``problog`` if installed through pip.

The CLI has different modes of operation. These can be accessed by adding a keyword as the first \
argument.

Currently, the following modes are supported

  * (default, no keyword): standard ProbLog inference
  * ``sample``: generate samples from a ProbLog program
  * ``mpe``: most probable explanation
  * ``lfi``: learning from interpretations
  * ``explain``: evaluate using mutually exclusive proofs
  * ``ground``: generate a ground program
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

    $ ./problog-cli.py some_heads.pl
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

    $ ./problog-cli.py sample some_heads.pl -N 3
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

    $ ./problog-cli.py sample some_heads.pl -N 3 --oneline
    % Probability: 0.2
    someHeads. % Probability: 0.2
    someHeads. % Probability: 0.3

By default, only query atoms are part of the sample.
To also include facts that were chosen while sampling, the argument ``--with-facts`` can be used.
The result above would then become

.. code-block:: prolog

    $ ./problog-cli.py sample some_heads.pl -N 3 --oneline --with-facts
    % Probability: 0.2
    heads1. someHeads. % Probability: 0.2
    heads2. someHeads. % Probability: 0.3

The sampling algorithm supports **evidence** through rejection sampling.  All generated samples \
are guaranteed to satisfy the evidence.  Note that this process can be slow if the evidence has \
low probability.

Sample based inference
++++++++++++++++++++++

It is also possible to use the sample mode for *probability estimation* by setting the flag \
``--estimate``.  The output is similar to the output in default mode.

The number of samples used for estimation can be determined in three ways:

    * by supplying the number of samples using the argument ``-N``
    * by supplying a timeout using the argument ``--timeout`` or ``-t`` (not supported on Windows)
    * by manually interrupting the process using CTRL-C or by sending a TERM(15) signal

.. code-block:: prolog

    $ ./problog-cli.py sample some_heads.pl  --estimate -t 5
    % Probability estimate after 7865 samples:
    someHeads : 0.79249841

References:
+++++++++++

    https://lirias.kuleuven.be/handle/123456789/510199


Most Probable Explanation (``mpe``)
-----------------------------------



Learning from interpretations (``lfi``)
---------------------------------------



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



Grounding (``ground``)
----------------------


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


Installation (``install``)
--------------------------



Web server (``web``)
--------------------



Testing (``unittest``)
----------------------



