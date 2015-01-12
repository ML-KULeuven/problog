%Expected outcome:
% a(1) 0.2
% a(2) 0.2
% a(3) 0.2

0.1::b(1).
0.2::b(2).
0.3::e(1).
0.4::e(3).

d(1).
d(2).
d(3).

a(X) :- b(2), c(X,Y).
c(X,Y) :- c(X,Z), c(Z,Y).
c(X,Y) :- d(X), d(Y).

query(a(X)).
