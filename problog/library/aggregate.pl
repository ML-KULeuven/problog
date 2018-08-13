:- use_module(library(lists)).
:- use_module(library(apply)).

aggregate(AggFunc, Var, Group, Body, (Group, Result)) :-
    all([Group, Body, Var], Body, L),
    maplist(nth0(0), L, LG),
    maplist(nth0(2), L, LV),
    enum_groups(LG, LV, Group, Values),
    call(AggFunc, Values, Result).


avg(L, Avg) :- sum(L, Sum), length(L, Count), Avg is Sum / Count.
sum(L, Sum) :- sum(L, 0, Sum).
min([H|T], Min) :- min(T, H, Min).
max([H|T], Max) :- max(T, H, Max).

sum([], Acc, Acc).
sum([X|T], Acc, Sum) :- Acc1 is Acc + X, sum(T, Acc1, Sum).

min([], Acc, Acc).
min([X|T], Acc, Min) :- X < Acc, min(T, X, Min).
min([X|T], Acc, Min) :- X >= Acc, min(T, Acc, Min).

max([], Acc, Acc).
max([X|T], Acc, Max) :- X >= Acc, max(T, X, Max).
max([X|T], Acc, Max) :- X < Acc, max(T, Acc, Max).
