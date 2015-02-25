Standard Grounding Engine
=========================

In order to transform a logic program into a ground program, ProbLog uses a Prolog-like engine. As the standard engine, we provide a simplified Prolog interpreter. This part of the documentation describes the operation of this standard engine.


Assumptions and restrictions
----------------------------

The grounding engine currently makes the following assumptions:

    1. All clauses in the program are *range-restricted*. This means that each query made to the engine should only have ground results. Some examples:
    
    .. code-block:: prolog
        
        r(A). r(B).
        q(A,B) :- A=B, r(A), r(B).      % not valid: the call A=B does not produce a ground result
        q(A,B) :- r(A), A=B, r(B).      % valid
        q(A,B) :- r(A), r(B), A=B.      % valid
            
    2. The program does not contain implicit unification between unbound variables.   
            
    3. The program is mostly *functor free*. Currently, functors are allowed in:

        * clause heads
        * call to arithmetic builtins, such as ``is/2`` and ``>/2``
        
    .. note ::
        There is no fundamental reason for this restriction, and it is scheduled to be removed.
    
    4. The engine only supports a limited number of *builtins*.

    .. note ::
        There is no fundamental reason for this restriction, more builtins will be added as the need arises.
        
    5. Programs should not contain *cycles with negation*.

    .. note ::
        There is no fundamental reason for this restriction, but the implementer currently has no clue about their semantics.

    
Implementation
--------------

The engine uses an event-based evaluation system, which is built around the notion of *process nodes*.

.. autoclass:: problog.engine.ProcessNode
    :members:


Key components
--------------

At the core of the engine is the node that processes definitions.
The correct operation of the engine relies on the following invariants:

    * There exists at most one ``ProcessDefine`` node for each call signature. A call signature consists of a predicate name and a list of arguments where all variables are anonymized. (singleton property)
    * The execute method of a ``ProcessDefine`` node is called at most once. (single run property)
    * All listeners of the ``ProcessDefine`` node receive *all* events, irrespective of when they register as a listener. (full disclosure property)
    * For each instantiation of its arguments the ``ProcessDefine`` node sends out exactly one event to each of its listeners. (no duplicates property)
    * The ``ProcessDefine`` node sends out its completion event exactly once to each of its listeners (no duplicates property)
    * The ``complete`` event is the last event that a node sends to a listener.


.. autoclass:: problog.engine.ProcessDefine
    :members:


Overview
--------

Evaluating a call
+++++++++++++++++

The operation of the engine is based on execution contexts. 
Every statement is executed within a certain context.

In the EventBasedEngine such a context is represented as a list of variable assignments.
Each variable that exists in the context corresponds to a certain position in the list.
Note that in this representation it is not possible to unify two non-instantiated variables (which would violated the range-restricted restriction for the ``'='/2`` call.)
A non-instantiated value is represented by ``None`` (we will use ``_`` in the examples).

Assume we are given a call ``p(X,f(Y,Z),c)``.
The following steps will be taken:

    1. The call arguments are instantiated from the context. For example, for context ``[x,_,g(z)]``, the call would become ``p(x,f(_,g(z)),c)``.
    
    2. The ``define`` node corresponding to the call predicate is found and each of its children is evaluated.
        
        * ``clause``
        * ``fact``
        
    

Implementation
--------------

.. automodule:: problog.engine
    :members:








