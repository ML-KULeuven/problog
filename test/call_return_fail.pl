%Expected outcome:
% p(1) 0.3

0.4::q(1,2).
0.3::q(1,1).
p(X) :- q(X,X).
query(p(X)).

