%Expected outcome:
% a 1
% b 0
% c 1
:-use_module(library(lists)).

even(X) :- X mod 2 =:= 0.

forall_even(L) :- forall(member(E,L), even(E)).

a :- forall_even([]).
b :- forall_even([1,2,3]).
c :- forall_even([2,48]).

query(a).
query(b).
query(c).