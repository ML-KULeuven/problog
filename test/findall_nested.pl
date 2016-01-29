%Expected outcome:
% q(4,[(1, [1]), (2, [1, 2]), (3, [1, 2, 3]), (4, [1, 2, 3, 4])])    1

q(N, List) :- findall((X,L), (between(1, N, X), findall(Y, between(1, X, Y), L)), List).

query(q(4, _)).