%%% Paste your program, query and evidence here

0.2::stressed(1).
0.2::stressed(X):-person(X).

person(1).
person(2).

query(stressed(1)).
query(stressed(2)).