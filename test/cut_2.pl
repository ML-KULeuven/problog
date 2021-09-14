%Expected outcome:
% ism(2,3,3)  1
% ism(2,3,2)  1
% ism(6,1,6)  1
% ism(6,1,1)  0

:- use_module(library(cut)).

% if X =< Y, then Z = Y, otherwise Z = X
m(1, X,Y,Z):- X =< Y, Z = Y.
m(2, X,Y,X).

ism(X, Y, Z) :- cut(m(X,Y,Z)).

query(ism(2,3,Z)).
query(ism(2,3,2)).
query(ism(6,1,6)).
query(ism(6,1,1)).
