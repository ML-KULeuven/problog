Writing models in ProbLog: basic concepts
=========================================

Tutorial
--------

See the `interactive tutorial <https://dtai.cs.kuleuven.be/problog/tutorial.html>`_.

Libraries
---------

ProbLog provides several libraries to simplify modelling.

Lists
+++++

See the `SWI-Prolog documentation <http://www.swi-prolog.org/pldoc/man?section=lists>`_ for a
description of these predicates.

.. code-block:: prolog

    :- use_module(library(lists)).

    member(Elem, List)
    select(Elem, List, Rest)
    select_uniform(ID, Values, Value, Rest)
    select_weighted(ID, Weights, Values, Value, Rest)
    select_weighted(ID, WeightsValues, Value, Rest)
    sum_list(List,Sum)
    max_list(List,Max)
    min_list(List,Min)
    unzip(ListAB,ListA,ListB)
    zip(ListA,ListB,ListAB)
    make_list(Len,Elem,List)
    append(ListA,ListB,ListAB)
    append(ListOfLists, List)
    prefix(Prefix,List)
    select(Elem1,List1,Elem2,List2)
    nth0(Index,List,Elem)
    nth1(Index,List,Elem)
    last(List,Last)
    reverse(L1,L2)
    permutation(List, Perm)

Apply
+++++

See the `SWI-Prolog documentation <http://www.swi-prolog.org/pldoc/man?section=apply>`_ for a
description of these predicates.

.. code-block:: prolog

    :- use_module(library(apply)).

    include(Goal, ListIn, ListYes)
    exclude(Goal, ListIn, ListNo)
    partition(Goal, ListIn, ListYes, ListNo)
    maplist(Goal, ListOut)
    maplist(Goal, List1, ListOut)
    maplist(Goal, List1, List2, ListOut)
    maplist(Goal, List1, List2, List3, ListOut)
    maplist(Goal, List1, List2, List3, List4, ListOut)
    foldl(Goal, List1, Start, Result)
    foldl(Goal, List1, List2, Start, Result)
    foldl(Goal, List1, List2, List3, Start, Result)
    foldl(Goal, List1, List2, List3, List4, Start, Result)
    scanl(Goal, List1, Start, ListOut)
    scanl(Goal, List1, List2, Start, ListOut)
    scanl(Goal, List1, List2, List3, Start, ListOut)
    scanl(Goal, List1, List2, List3, List4, Start, ListOut)
