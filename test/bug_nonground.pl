%Expected outcome:
%  q(1,2)        0.784
%  q(1)         0.3

0.4::a(1).
0.5::a(2).
0.3::b(1).
0.6::b(2).

0.1::c(1,2).
0.3::c(2,1).

p(X, Y) :- a(X).
p(X, Y) :- b(Y).
p(X, Y) :- c(X, Y).

%query(p(_,_)).

q(Y) :- p(X, Y), X = 3.

query(q(1)).

q(X, Y) :- p(X, Y).

query(q(1,2)).
