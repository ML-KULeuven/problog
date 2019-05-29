normal(B,1)~a:- c, b~=B.
normal(A,1)~b:- d, a~=A.
normal(0,1)~a:- d.
normal(0,1)~b:- c.
0.5::c.
0.5::d.

test:- a~=A, conS(A>0).
query(test).
