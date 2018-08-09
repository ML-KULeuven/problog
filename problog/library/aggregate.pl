:- use_module(library(lists)).


groupby([], []).
groupby([(G,X,V)|T], Out) :-
    groupby(T, G, [V], Out).


% groupby(ListIn, CurEl, CurGroup, Out)
groupby([], G, S, [(G, S)]).

groupby([(G,X,V)|T], G, S,  Out) :-
    groupby(T, G, [V|S], Out).

groupby([(G,X,V)|T], G1, S,  [(G1,S)|Out]) :-
    G \= G1,
    groupby(T, G, [V], Out).


aggregate(AggFunc, Var, Group, Body, (Group, Result)) :-
    all((Group, Body, Var), Body, L),
    sort(L, Ls),
    groupby(Ls, Groups),
    member((Group, Values), Groups),
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
