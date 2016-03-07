Relation to Prolog
==================

ProbLog supports a subset of the Prolog language for expressing models in probabilistic logic.
The main difference between ProbLog's language and Prolog is that Prolog is a complete logic programming language, 
whereas ProbLog is a logic representation language.
This means that most of the functionality of Prolog that is related to the programming part (such as control constructs and input/output) are not supported in ProbLog.

The list of supported builtins is based on Yap Prolog. See section 6 of the Yap manual for an explanation of these predicates.

Control predicates
++++++++++++++++++

**Supported:**

 * ``P, Q``
 * ``P; Q``
 * ``true/0``
 * ``fail/0``
 * ``false/0``
 * ``\+/1``
 * ``not/1``
 * ``call/1``
 * ``call/N`` (for N up to 9)
 
**Not supported:**
 
 * ``!/0``
 * ``P -> Q``
 * ``P *-> Q``
 * ``repeat``
 * ``incore/1`` (use ``call/1``)
 * ``call_with_args/N`` (use ``call/N``)
 * ``P`` (use ``call/1``)
 * ``if(A,B,C)`` (use ``(A,B);(\+A,C)``)
 * ``once/1``
 * ``forall(A,B)`` (use ``\+(A,\+B)``)
 * ``ignore/1`` 
 * ``abort/0``
 * ``break/0``
 * ``halt/0``
 * ``halt/1``
 * ``catch/3``
 * ``throw/1``
 * ``garbage_collect/0``
 * ``garbage_collect_atoms/0``
 * ``gc/0``
 * ``nogc/0``
 * ``grow_heap/1``
 * ``grow_stack/1``
 
**To be added:** 

 * ``forall/2``
 * ``if/3``
 
Handling Undefined Procedures
+++++++++++++++++++++++++++++

**Alternative:**

 * ``unknown(fail)`` can be used

**Not supported:** all


Message Handling
++++++++++++++++

**Not supported:** all

Predicates on Terms
+++++++++++++++++++

**Supported:**

 * ``var/1``
 * ``atom/1``
 * ``atomic/1``
 * ``compound/1``
 * ``db_reference/1`` (always fails)
 * ``float/1``
 * ``rational/1`` (always fails)
 * ``integer/1``
 * ``nonvar/1``
 * ``number/1``
 * ``primitive/1``
 * ``simple/1``
 * ``callable/1``
 * ``ground/1``
 * ``arg/3``
 * ``functor/3`` 
 * ``T =.. L``
 * ``X = Y``
 * ``X \= Y``
 * ``is_list/1``
 
**Not supported:**

 * ``numbervars/3``
 * ``unify_with_occurs_check/2``
 * ``copy_term/2``
 * ``duplicate_term/2``
 * ``T1 =@= T2``
 * ``subsumes_term/2``
 * ``acyclic_term/1``
 
Predicates on Atoms
+++++++++++++++++++

**Not supported:** all

**To be added:** all

Predicates on Characters
++++++++++++++++++++++++

**Not supported:** all

**To be added:** all

Comparing Terms
+++++++++++++++

**Supported:**

 * ``compare/3``
 * ``X == Y`` (not supported for two variables)
 * ``X \== Y`` (not supported for two variables)
 * ``X @< Y``
 * ``X @=< Y`` (all variables are considered equal)
 * ``X @< Y``
 * ``X @> Y``
 * ``X @>= Y`` (all variables are considered equal)
 * ``sort/2`` (all variables are considered equal, e.g. ``sort([X,Y,Y],S)`` returns ``S=[_]`` where Prolog would return ``S=[X,Y]`` or ``S=[Y,X]``).
 * ``length/2`` (both arguments unbound not allowed)
 
**Not supported:**

 * ``keysort/2``
 * ``predsort/2``
 
Arithmetic
++++++++++

**Supported:**
 
 * ``X``
 * ``-X``
 * ``X+Y``
 * ``X-Y``
 * ``X*Y``
 * ``X/Y``
 * ``X//Y``
 * ``X mod Y``
 * ``X rem Y`` (currently same as mod)
 * ``X div Y``
 * ``exp/1``
 * ``log/1``
 * ``log10/1``
 * ``sqrt/1``
 * ``sin/1``
 * ``cos/1``
 * ``tan/1``
 * ``asin/1``
 * ``acos/1``
 * ``atan/1``
 * ``atan/2``
 * ``sinh/1``
 * ``cosh/1``
 * ``tanh/1``
 * ``asinh/1``
 * ``acosh/1``
 * ``atanh/1``
 * ``lgamma/1``
 * ``erf/1``
 * ``erfc/1``
 * ``integer/1``
 * ``float/1``
 * ``float_fractional_part/1``
 * ``float_integer_part/1``
 * ``abs/1``
 * ``ceiling/1``
 * ``floor/1``
 * ``round/1``
 * ``sign/1``
 * ``truncate/1``
 * ``max/2``
 * ``min/2``
 * ``X ^ Y``
 * ``exp/2``
 * ``X ** Y``
 * ``X /\ Y``
 * ``X \/ Y``
 * ``X # Y``
 * ``X >< Y``
 * ``X xor Y``
 * ``X << Y``
 * ``X >> Y``
 * ``\ X``
 * ``pi/0``
 * ``e/0``
 * ``epsilon/0``
 * ``inf/0``
 * ``nan/0``
 * ``X is Y``
 * ``X < Y``
 * ``X =< Y``
 * ``X > Y``
 * ``X >= Y``
 * ``X =:= Y``
 * ``X =\= Y``
 * ``between/3``
 * ``succ/2``
 * ``plus/3``
 
**Not supported:**
 
 * ``random/1``
 * ``rational/1``
 * ``rationalize/1``
 * ``gcd/2``
 * ``msb/1``
 * ``lsb/1``
 * ``popcount/1``
 * ``[X]``
 * ``cputime/0``
 * ``heapused/0``
 * ``local/0``
 * ``global/0``
 * ``random/0``
 * ``srandom/1``
 
Remaining sections
++++++++++++++++++

**Not supported:** all


