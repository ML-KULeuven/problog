beta(1,1)~a:- d.
beta(C,1)~a:- \+d, c~=C.

beta(A,1)~b:- d, e, a~=A.
beta(A,B)~c:- d, e, a~=A, b~=B.

beta(1,1)~b:- d, \+e.
beta(A,1)~c:- d, \+e, a~=A.

beta(C,1)~b:- \+d, e, c~=C.
beta(1,1)~c:- \+d, e.

0.5::e.
0.5::d.

t:- c~=C, conS(C>1/2).

query(t).
