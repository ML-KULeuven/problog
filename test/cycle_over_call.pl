%Expected outcome:
% a(1) 1
% a(2) 1

a(1).
a(2).

a(X) :- call(a(X)).

query(a(L)).