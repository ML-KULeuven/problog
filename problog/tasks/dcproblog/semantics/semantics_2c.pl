normal(B,1)~a:- b~=B.
normal(A,1)~b:- a~=A.
normal(0,1)~a.
normal(0,1)~b.

test:- a~=A, conS(A>0).
query(test).
