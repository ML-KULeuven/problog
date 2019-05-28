%Expected outcome:
% scope(3):a 1
% scope(3):b 1
% scope(3):(a, b) 1
% [scope(1), scope(2)]:b 1
% scope(1):c 1


:- use_module(library(scope)).

scope(1):a.
scope(2):b.
scope(3):X :- scope(1):X; scope(2):X.
c.

query(scope(3):_).

query(scope(3):(a,b)).

query([scope(1),scope(2)]:b).

query(scope(1):c).
