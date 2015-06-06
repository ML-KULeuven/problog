%Expected outcome:
% p(1,1) 1

a(1,_).
a(2,_).

p(X,Y) :- a(X,Z1), a(Y,Z2), Z1=Z2, Z1 is X+Y, Z2 = 2.

query(p(X,Y)).