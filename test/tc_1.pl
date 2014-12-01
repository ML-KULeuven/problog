%Expected outcome: 
% stressed(1) 0.36
% stressed(2) 0.2

0.2::stressed(1).
0.2::stressed(X):-person(X).

person(1).
person(2).

query(stressed(1)).
query(stressed(2)).