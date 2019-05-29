%Expected outcome:
% -----
% b(1) 1.0
% -----
% b(2) 1.0

a(1).
a(2).
?::b(X) :- a(X).

ensure_false :- b(X), b(Y), X \= Y.

query(b(_)).