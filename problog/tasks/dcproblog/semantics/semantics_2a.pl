a~normal(B,1) :- c, B is b.
b~normal(A,1) :- \+c, A is a.
a~normal(0,1) :- \+c.
b~normal(0,1) :- c.
0.5::c.
test:- a>0.
query(test).
