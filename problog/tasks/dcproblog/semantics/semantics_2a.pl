normal(B,1)~a :- c, b~=B.
normal(A,1)~b :- \+c, a~=A.
normal(0,1)~a :- \+c.
normal(0,1)~b :- c.
0.5::c.
test:- a~=A, conS(A>0).
query(test).
