%Expected outcome:
% ERROR NonGroundProbabilisticClause

0.4::b(1).
0.4::b(2).
0.4::c(1).
0.4::c(2).

0.4 :: a(X,Y) :- \+b(X), \+c(Y).


query(a(X,Y)).