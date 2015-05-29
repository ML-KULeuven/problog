ProbLog 2.1 command line interface
==================================

The command line interface (CLI) gives access to the basic functionality of ProbLog 2.1.
It is accessible through the script ``problog-cli.py``.

The CLI has different modes of operation. These can be accessed by adding a keyword as the first \
argument.

Currently, the following modes are supported

  * (default, no keyword): standard ProbLog inference
  * ``sample``: generate samples from a ProbLog program
  * ``ground``: generate a ground program
  * ``install``: run the installer
  * ``unittest``: run the testsuite

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


... to complete ...

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



Grounding (``ground``)
----------------------



Installation (``install``)
--------------------------



Testing (``unittest``)
----------------------



