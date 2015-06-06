%Expected outcome:
% a(1,1) 1
% p(1) 1

a(2,3).
a(1,1).

p(X) :- a(X,X).

query(a(X,X)).
query(p(X)).


