:- module(builtin, [forall/2]).

forall(A, B) :- \+(call(A), \+call(B)).