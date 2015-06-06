Introduction
============

Overview
--------

LP -> Grounder -> GP -> Compiler -> CP -> Evaluator

LogicProgram (LP)
+++++++++++++++++

Interface:
  * __iter__ -> ( Clause / And / Or / Not / Lit )
  * addClause( Clause / And / Or / Not / Lit ) -> whatever
  * __str__  -> valid Pro(b)Log code

Examples are:
 * PrologFile (filename)
 * ClauseDB
 * ...
 
Grounder
++++++++

Interface:
  * ground( LP ) -> GP
  * prepare( LP ) -> LP         (transform LP to optimal format)
  * needsPrepare( LP ) -> bool  (check whether conversion is needed)

prepare() is done automatically in ground() if necessary

Examples are:
  * Yap grounder (current)
  * Python grounder (under development)
  * TP-based grounder (Jonas)
  * ...

GroundProgram (GP)
++++++++++++++++++

Basically an AND-OR graph.

Interface:
  * TBD 

Examples:
  * ProbFOIL's Grounding
  * CNF
  * ...
 

Compiler
++++++++

Interface: (similar to Grounder)
  * compile( GP ) -> CP
  * prepare( GP ) -> GP
  * needsPrepare( GP ) -> bool

Examples are:
  * d-DNNF (c2d, dsharp)
  * SDD
  * BDD
  * ...
  
CompiledProgram (CP)
++++++++++++++++++++

Same structure as GroundProgram? Also an AND-OR graph.

Interface:
  * TBD

Evaluator
+++++++++

Interface: 
  * TBD

