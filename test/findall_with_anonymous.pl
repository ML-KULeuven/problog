%Expected outcome:
% all_p([1, 1, 2, 2, 3, 3]) 0.4

p(X, Y) :- between(1,3,X), q(Y).
q(a).
q(b).

0.4::all_p(L) :- findall(X, p(X,_), L).

query(all_p(_)). 