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
  * ``map``: MAP inference
  * ``explain``: evaluate using mutually exclusive proofs
  * ``ground``: generate a ground program
  * ``bn``: export a Bayesian network
  * ``shell``: interactive shell
  * ``install``: run the installer
  * ``unittest``: run the testsuite
  * ``web``: start a web server

Default (no keyword)
--------------------

Run ProbLog in standard inference mode.

Used as ``problog <model> [optional]`` where:

- ``<model>`` is a file containing the ProbLog model;
- ``[optional]`` is a set of optional parameters.

Returns a set of probabilities for the queries in the model.


For example, given a file ``some_heads.pl``

.. code-block:: prolog

    $ cat some_heads.pl
    0.5::heads1.
    0.6::heads2.
    someHeads :- heads1.
    someHeads :- heads2.

    query(someHeads).

We can do

.. code-block:: shell

    $ problog some_heads.pl
    someHeads : 0.8

This mode supports many optional arguments to customize the inference:


- ``--knowledge {sdd,sddx,bdd,nnf,ddnnf,kbest,fsdd,fbdd}, -k {sdd,sddx,bdd,nnf,ddnnf,kbest,fsdd,fbdd}``;  Knowledge compilation tool. By default, it uses the first available option from SDD, d-DNNF using c2d and d-DNNF using dsharp.
- ``--combine``             Combine input files into single model.
- ``--logspace``            Use log space evaluation (default).
- ``--nologspace``          Use normal space evaluation.
- ``--symbolic``            Use symbolic computations.
- ``--output OUTPUT, -o OUTPUT``; Output file (default stdout)
- ``--recursion-limit RECURSION_LIMIT``; Set Python recursion limit. (default: 10000)
- ``--timeout TIMEOUT, -t TIMEOUT``; Set timeout (in seconds, default=off).
- ``--compile-timeout COMPILE_TIMEOUT``; Set timeout for compilation (in seconds, default=off).
- ``--debug, -d``           Enable debug mode (print full errors).
- ``--full-trace, -T``      Full tracing.
- ``-a ARGS, --arg ARGS``   Pass additional arguments to the cmd_args builtin.
- ``--profile``            output runtime profile
- ``--trace``               output runtime trace
- ``--profile-level PROFILE_LEVEL``
- ``--format {text,prolog}``
- ``-L LIBRARY, --library LIBRARY``; Add to ProbLog library search path
- ``--propagate-evidence``;  Enable evidence propagation
- ``--dont-propagate-evidence``; Disable evidence propagation
- ``--propagate-weights``;   Enable weight propagation
- ``--convergence CONVERGENCE, -c CONVERGENCE``; Stop anytime when bounds are within this range


Sampling (``sample``)
--------------------------------------------------


Run ProbLog in sampling mode, generating possible assignments to the queries in the model.

Used as ``problog sample <model> [optional]`` where:

- ``<model>`` is a file containing the ProbLog model with the queries of interest;
- ``[optional]`` is a set of optional parameters.

For example, given a file ``some_heads.pl``

.. code-block:: prolog

    $ cat some_heads.pl
    0.5::heads1.
    0.6::heads2.
    someHeads :- heads1.
    someHeads :- heads2.

    query(someHeads).

We can do:

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


By default, only query atoms are part of the sample. To also include facts that were chosen while sampling, the argument ``--with-facts`` can be used.
The result above would then become

.. code-block:: prolog

    $ problog sample some_heads.pl -N 3 --oneline --with-facts
    % Probability: 0.2
    heads1. someHeads. % Probability: 0.2
    heads2. someHeads. % Probability: 0.3

The sampling algorithm supports **evidence** through rejection sampling.  All generated samples
are guaranteed to satisfy the evidence.  Note that this process can be slow if the evidence has
low probability.

The sampling algorithm support evidence propagation, that is, in certain cases it can ensure the
evidence holds without the use of rejection sampling.
To enable this feature use the ``--propagate-evidence`` argument. Evidence propagation is not
supported on programs with continuous distributions, or on programs where the evidence has
infinite support.


All the optional arguments:

- ``-h, --help``; show the help message and exit
- ``-N N, -n N``;Number of samples.
- ``--with-facts``; Also output choice facts (default: just queries).
- ``--with-probability``; Show probability.
- ``--as-evidence``; Output as evidence.
- ``--propagate-evidence``; Enable evidence propagation
- ``--dont-propagate-evidence``; Disable evidence propagation
- ``--oneline``; Format samples on one line.
- ``--estimate``; Estimate probability of queries from samples (see next section).
- ``--timeout TIMEOUT, -t TIMEOUT``; Set timeout (in seconds, default=off).
- ``--output OUTPUT, -o OUTPUT``; Filename of output file.
- ``--verbose, -v``; Verbose output
- ``--seed SEED, -s SEED``; Random seed
- ``--full-trace``;
- ``--strip-tag``; Strip outermost tag from output.
- ``-a ARGS, --arg ARGS``; Pass additional arguments to the cmd_args builtin.
- ``--progress``; show progress.






Sample based inference
++++++++++++++++++++++

The sample mode can be used for *probability estimation* by setting the flag \
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

    Paper: https://lirias.kuleuven.be/handle/123456789/510199

    Tutorial: https://dtai.cs.kuleuven.be/problog/tutorial/sampling/02_arithmeticexpressions.html

Most Probable Explanation (``mpe``)
-----------------------------------



Run ProbLog in MPE mode, computing the possible world with the highest probability in which all queries
and evidence are true.


Used as ``problog mpe <model> [optional]`` where:

- ``<model>`` is a file containing the ProbLog model;
- ``[optional]`` is a set of optional parameters.

Returns:

- the possible world with the highest probability (as a set of facts);
- the probability of the most probable explanation.

The optional arguments are:

- ``-h, --help``; show this help message and exit
- ``--solver {maxsatz,scip,sat4j}``;  MaxSAT solver to use
- ``--full``; Also show false atoms.
- ``-o OUTPUT, --output OUTPUT``;  Write output to given file (default: write to stdout)
- ``-v, --verbose``; Increase verbosity

For example, given a file ``digraph.pl`` describing a probabilistic graph:

.. code-block:: shell

    $ cat digraph.pl
    0.6::edge(1,2).
    0.1::edge(1,3).
    0.4::edge(2,5).
    0.3::edge(2,6).
    0.3::edge(3,4).
    0.8::edge(4,5).
    0.2::edge(5,6).

    path(X,Y) :- edge(X,Y).
    path(X,Y) :- edge(X,Z), Y \== Z,path(Z,Y).

    evidence(path(1,5)).
    evidence(path(1,6)).

We can do:

.. code-block:: shell

    $ problog mpe pgraph.pl
    edge(4,5)  edge(1,2)  edge(2,5) edge(2,6)
    \+edge(1,3)  \+edge(3,4)  \+edge(5,6)
    % Probability: 0.0290304


Learning from interpretations (``lfi``)
---------------------------------------
Run ProbLog in the learning from interpretation (LFI) setting. Given a probabilistic program with parameterized weights
and a set of (partial) interpretation, learns appropriate values of the parameters.

Used as: ``problog lfi <model> <evidence> [optional]`` where:

- ``<model>`` is the ProbLog model file;
- ``<evidence>`` is the a file containing a set of examples to learn from.
- ``[optional]`` are optional arguments

The command standard output is: ``<loss> <probs> <atoms> <iter>`` where:

- ``<loss>`` is the final loss of the learning problem;
- ``<probs>`` is a list of the learned paramenters (i.e. probabilities);
- ``<atoms>`` is the list of clauses that the probabilities refer to (positional mapping);
- ``<iter>`` is the number of EM iterations.

The optional arguments are:

- ``-h, --help``; show the help message and exit
- ``-n MAX_ITER``;
- ``-d MIN_IMPROV``;
- ``-O OUTPUT_MODEL, --output-model OUTPUT_MODEL``;  write resulting model to given file
- ``-o OUTPUT, --output OUTPUT``; write output to file
- ``--logger LOGGER``; write log to a given file
- ``-k {sdd,sddx,ddnnf}, --knowledge {sdd,sddx,ddnnf}``; knowledge compilation tool
- ``--logspace``; use log space evaluation
- ``-l LEAKPROB, --leak-probabilities LEAKPROB``; Add leak probabilities for evidence atoms.
- ``--propagate-evidence``; Enable evidence propagation
- ``--dont-propagate-evidence``; Disable evidence propagation
- ``--normalize``; Normalize AD-weights.
- ``-v, --verbose``;
- ``-a ARGS, --arg ARGS``;   Pass additional arguments to the cmd_args builtin.


An example of model file ``some_heads.pl``:

.. code-block:: prolog

    t(_)::heads1.
    t(_)::heads2.
    someHeads :- heads1.
    someHeads :- heads2.

An example of evidence file ``some_heads.pl``:

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

An example of LFI call:

.. code-block:: shell

    $ problog lfi some_heads.pl some_heads_ev.pl -O some_heads_learned.pl
    -1.7917594692732088 [0.33333333, 0.5] [t(_)::heads1, t(_)::heads2] 21

The learned program is saved in ``some_heads_learned.pl``.

.. code-block:: shell

    $ cat some_heads_learned.pl
    0.33333333::heads1.
    0.5::heads2.
    someHeads :- heads1.
    someHeads :- heads2.




Decision Theoretic ProbLog (``dt``)
-----------------------------------

Run ProbLog in decision-theoretic mode.

Used as: ``problog dt <model> [optional]`` where:

- ``<model>`` is the a decision-theoretic ProbLog model file;
- ``[optional]`` are optional arguments

The command standard output is ``<choices> <score>`` where:

- ``<choices>`` are the best decisions;
- ``<scores>`` is the score for the best decision.

The current implementation supports two evaluation strategies: exhaustive search (exact) and local search (approximate).
Exhaustive search is the default. Local search can be enabled with the argument ``-s local``.

The optional arguments are:

- ``-h, --help``; show the help message and exit
- ``--knowledge {sdd,sddx,bdd,nnf,ddnnf,kbest,fsdd,fbdd}, -k {sdd,sddx,bdd,nnf,ddnnf,kbest,fsdd,fbdd}``; Knowledge compilation tool.
- ``-s {local,exhaustive}``; --search {local,exhaustive}
- ``-v, --verbose``; Set verbosity level
- ``-o OUTPUT, --output OUTPUT``;  Write output to given file (default: write to stdout)


For example, given the DT-model:

.. code-block:: shell

    $ cat dt_model.pl
    0.3::rain.
    0.5::wind.
    ?::umbrella.
    ?::raincoat.
    broken_umbrella :- umbrella, rain, wind.
    dry :- rain, raincoat.
    dry :- rain, umbrella, not broken_umbrella.
    dry :- not(rain).
    utility(broken_umbrella, -40).
    utility(raincoat, -20).
    utility(umbrella, -2).
    utility(dry, 60).


we can do:
    .. code-block:: shell

        $ problog dt dt_model.pl
        raincoat:	0
        umbrella:	1
        SCORE: 43.00000000000001


References:

    https://lirias.kuleuven.be/handle/123456789/270066


MAP inference (``map``)
-----------------------

Run ProbLog in MAP mode. Only facts that occur as explicit queries are assigned and all other probabilistic facts are marginalized over.
MAP inference is implemented on top of DT-ProbLog.

Used as: ``problog map <model> [optional]`` where:

- ``<model>`` is the a ProbLog model file;
- ``[optional]`` are optional arguments

The command standard output is ``<choices> <score>`` where:

- ``<choices>`` are the MAP assignments;
- ``<scores>`` is the score for the MAP.

The current implementation supports two evaluation strategies: exhaustive search (exact) and local search (approximate).
Exhaustive search is the default. Local search can be enabled with the argument ``-s local``.

The optional arguments are:

- ``-h, --help``; show the help message and exit
- ``--knowledge {sdd,sddx,bdd,nnf,ddnnf,kbest,fsdd,fbdd}, -k {sdd,sddx,bdd,nnf,ddnnf,kbest,fsdd,fbdd}``; Knowledge compilation tool.
- ``-s {local,exhaustive}``; --search {local,exhaustive}
- ``-v, --verbose``; Set verbosity level
- ``-o OUTPUT, --output OUTPUT``;  Write output to given file (default: write to stdout)


Explanation mode (``explain``)
------------------------------

Run ProbLog in explain mode.

Used as: ``problog explain <model> [optional]``.


The ``explain`` mode offers insight in how probabilities can be computed for a ProbLog program.
Given a model, the output consists of three parts:

  * a reformulation of the model in which annotated disjunctions and probabilistic clauses are rewritten
  * for each query, a list of mutually exclusive proofs with their probability
  * for each query, the success probability determined by taking the sum of the probabilities of the individual proofs

This mode currently does not support evidence.



Grounding (``ground``)
----------------------

Run ProbLog ground routine.

Used as: ``problog ground <model> [optional]``.

The ``ground`` mode provides access to the ProbLog grounder.
Given a model, the output consists of the ground program.


The optional arguments are:
- ``-h, --help``; show the help message and exit
- ``--format {dot,pl,cnf,svg,internal}``; output format. The output can be formatted in different formats:
  * pl: ProbLog format
  * dot: GraphViz representation of the AND-OR tree
  * svg: GraphViz representation of the AND-OR tree as SVG (requires GraphViz)
  * cnf: DIMACS encoding as CNF
  * internal: Internal representation (for debugging)
- ``--break-cycles``; perform cycle breaking
- ``--transform-nnf``; transform to NNF
- ``--keep-all``; also output deterministic nodes
- ``--keep-duplicates``; don't eliminate duplicate literals
- ``--any-order``; allow reordering nodes
- ``--hide-builtins``; hide deterministic part based on builtins
- ``--propagate-evidence``; propagate evidence
- ``--propagate-weights``; propagate evidence
- ``--compact``; allow compact model (may remove some predicates)
- ``--noninterpretable``;
- ``--verbose, -v``; Verbose output
- ``-o OUTPUT, --output OUTPUT``; output file
- ``-a ARGS, --arg ARGS``; Pass additional arguments to the cmd_args builtin.


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

