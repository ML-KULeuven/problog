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
    
     
    