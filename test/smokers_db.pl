%Expected outcome:
% smokes(bob)   0.73289308


:- use_module(library(sqlite)).

:- sqlite_load('smokers.sqlite').

P :: influences(X, Y) :- friend_of(X, Y, P).

0.3::smokes(X) :- person(X).

smokes(X) :- influences(Y,X), smokes(Y).

query(smokes(bob)).


