Notes
=====

.. warning::

	These notes are for Anton's internal use only.

Data structures
---------------

ProbLog revolves around data structures for storing logic formulas.
All these formulas consist of the four components And, Or, Not and Literal.

The following structures occur:

  Logic Program
      a first-order directed and-or-graph

  Ground Program
      a *propositional* directed and-or-graph
      
  Acyclic Ground Program
      a propositional directed *acyclic* and-or-graph

  Negation Normal Form
      a propositional directed *acyclic* and-or-graph where negation is only applied to literals
      
  Conjunctive Normal Form (CNF) (subset of NNF)
      a two-layered NNF where the top layer is a conjunction and the second layer contains disjunctions
      
  Deterministic Decomposable NNF (d-DNNF)
      a NNF that is deterministic and decomposable (see below)
      
  Smooth Deterministic Decomposable NNF (sd-DNNF)
      a NNF that is deterministic, decomposable and smooth (see below)
     
  Ordered Binary Decision Diagram (OBDD)
      another type of NNF 
      
  Sentential Decision Diagram (SDD)
      another type of NNF
      
The last four support efficient weighted model counting (WMC).
      
For NNFs we can observe the following properties:

  Deterministic
      Disjuncts are logically disjoint.
  
  Decomposable
      Conjuncts do not share variables.
      
  Smoothness
      All disjuncts mention the same set of variables.
    
.. graphviz::

  digraph foo {
    LP [ label="Logic Program"];
    GP [ label="Ground Program"];
    AGP [ label="Acyclic Ground Program"];
    CNF [ label="CNF"];
    dDNNF [ label="d-DNNF"];
    sdDNNF [ label="sd-DNNF"];
    OBDD [ label="OBDD"];
    SDD [ label="SDD"];  
    LP -> GP [ label="ground"];
    GP -> AGP [ label="loop-breaking"];
    AGP -> CNF [ label="cnf conversion" ];
    CNF -> dDNNF [ label="c2d / dsharp"];
    dDNNF -> sdDNNF [ label="smoothen"];
    CNF -> sdDNNF [ label="c2d / dsharp"];
    CNF -> OBDD [ label="cudd" ];
    CNF -> SDD [ label="libSDD" ];
  }
  
  
Annotated disjunctions
----------------------

Overview
++++++++

An annotated disjunction is a clause with multiple heads.
When the body of the clause is true, *one of the heads* is selected according to the given probabilities.
An example of an annotated disjunction is the following:

.. code-block:: prolog

    p1::a(X); p2::b(Y); p3::c(X,Y) <- p(X,Z), p(Z,Y).

This clause can be rewritten into it's equivalent program:
    
.. code-block:: prolog

    p12::p(1,2).
    p23::p(2,3).
    p14::p(1,4).
    p43::p(4,3).
    p34::p(3,4).
    
    adX_clause(X,Y) :- p(X,Z), p(Z,Y).
    
    (pA,1)::choose_aX(_,_).
    (pB,1)::choose_bY(_,_).
    (pC,1)::choose_cXY(_,_).
    
    a(X)   :- clause_body(X,Y), choose_aX(X,Y).
    b(Y)   :- clause_body(X,Y), choose_bY(X,Y).
    c(X,Y) :- clause_body(X,Y), choose_cXY(X,Y).

    mutually_exclusive( choose_aX, choose_bY, choose_cXY ).
    
ClauseDB
++++++++

The example annotated disjunction can be represented in a compiled format as:

.. code-block:: python

    0: clause( functor='adX_clause', args=(0,1), child=1, varcount=3 )   # body
    1: conj( children=(2,3) )
    2: call( functor='p', args=(0,2), defnode=4 )
    3: call( functor='p', args=(2,1), defnode=4 )
    4: define( functor='p', arity=2, children=[...] )
    5: choice( functor='adX_choice_0', args=(0,1), probability=pA, group=X )
    6: choice( functor='adX_choice_1', args=(0,1), probability=pB, group=X )    
    7: choice( functor='adX_choice_2', args=(0,1), probability=pC, group=X )
    8: clause( functor='a', args=(0,), child=11 )
    9: clause( functor='b', args=(1,), child=12 )
   10: clause( functor='c', args=(0,1,), child=13 )
   11: conj( children=(14,16) )      => should be call -> choice
   12: conj( children=(14,17) )      => should be call -> choice
   13: conj( children=(14,18) )      => should be call -> choice
   14: call( functor='adX_clause', args=(0,1), defnode=0 )
   15: define( functor='adX_clause', arity=2, children=[0] )
   16: call( functor='adX_choice_0, args=(0,1), defnode=5 )
   17: call( functor='adX_choice_1, args=(0,1), defnode=6 )
   18: call( functor='adX_choice_2, args=(0,1), defnode=7 )      
   
   ..: define( functor='a', arity=1, children=8 )
   ..: define( functor='b', arity=1, children=9 )
   ..: define( functor='c', arity=2, children=8 )
   ..: ... facts for p/2 ...
   
.. note::
      
      A ``def`` node can be skipped if there is only one definition.
      
Grounding
+++++++++

During grounding the annotated a choice node has to be created for each ground substitution:

.. code-block:: prolog

    p12::p(1,2).
    p23::p(2,3).
    p14::p(1,4).
    p43::p(4,3).
    p34::p(3,4).
        
    adX_clause_body(1,3). % [ 12+23 / 14+43 ]
    adX_clause_body(4,4). % [ 43+34 ]
    adX_clause_body(2,4). % [ 23+34 ] 
    
    % Group X_(1,3)
    (pA,1)::choose_aX(1,3).
    (pB,1)::choose_bY(1,3).
    (pC,1)::choose_cXY(1,3).
    a(1)   :- adX_clause_body(1,3), choose_aX(1,3).
    b(3)   :- adX_clause_body(1,3), choose_bY(1,3).
    c(1,3) :- adX_clause_body(1,3), choose_cXY(1,3).

    % Group X_(4,4)
    (pA,1)::choose_aX(4,4).
    (pB,1)::choose_bY(4,4).
    (pC,1)::choose_cXY(4,4).
    a(4)   :- adX_clause_body(4,4), choose_aX(4,4).
    b(4)   :- adX_clause_body(4,4), choose_bY(4,4).
    c(4,4) :- adX_clause_body(4,4), choose_cXY(4,4).
    
    % Group X_(2,4)
    (pA,1)::choose_aX(2,4).
    (pB,1)::choose_bY(2,4).
    (pC,1)::choose_cXY(2,4).
    a(2)   :- adX_clause_body(2,4), choose_aX(2,4).
    b(4)   :- adX_clause_body(2,4), choose_bY(2,4).
    c(2,4) :- adX_clause_body(2,4), choose_cXY(2,4).
   
Note that ``b(4)`` occurs for two of the body groundings. The nodes ``choose_`` are mutually exclusive within the same group.

   
..
    Annotated disjunction are clauses with multiple literals in the head. Each of these literals has a probability which indicates the probability that this head is true because of this rule. The head literals are mutually exclusive, that is, only one can be chosen. As a result, their sum of probabilities should be less or equal to 1.

    There are two ways of dealing with annotated disjunctions:

      1. by rewriting the model where the heads are replaced by a sequence of heads
      2. by adding an explicit constraint that states the heads are mutually exclusive
  
    The first solution is not correct for all inference tasks, for example, most probable explanation (MPE).

    Given the following model:

    .. code-block:: prolog

      p11::a1, p12::a2, p13::a3          <- body1.
      p21::a1, p22::a2,        , p23::a4 <- body2. 

    We can introduce literals for each head-body combination.

    .. code-block:: prolog

      a1 :- ad1_a1, body1.
      a1 :- ad2_a1, body2.
      a2 :- ad1_a2, body1.
      a2 :- ad2_a2, body2.
      a3 :- ad1_a3, body1.
      a4 :- ad1_a4, body2.
  
      p11::ad1_a1.
      p12::ad1_a2.
      p13::ad1_a3.
  
      p21::ad2_a1.
      p22::ad2_a2.
      p23::ad2_a4.
  
    This model is not equivalent to the previous one, because it does not take into account the mutual exclusivity of (p11,p12,p13) and (p21,p22,p23).
    In order to incorporate this restriction we need to modify the model in two ways:

    	* add a constraint that states that p11, p12 and p13 are mutually exclusive.
    	* set the probability of ~ad1_a1 and others to 1 (because through mutually exlusivity the choice of *not* selecting ad1_a1 is completely determined by the choice for one of the other alternatives.)

    Adding annotated disjunctions thus requires support for these two things:

    	* additional logic constraints (e.g. mutual exclusivity)
    	* separate positive and negative probabilities (where :math:`p_\ominus \not = 1-p_\oplus`)
	

    .. code-block:: prolog

        p11::a1, p12::a2, p13::a3 <- b, c. 

        ===
    
        body <=> b, c
    
        not (ad1_a1 /\ ad1_a2) and not (ad1_a1 /\ ad1_a3) and not (ad1_a2 /\ ad1_a3) and (not body \/ ad1_a1 \/ ad1_a2 \/ ad1_a3)
    


    