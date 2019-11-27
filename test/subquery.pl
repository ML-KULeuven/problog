%Expected outcome:
% b(0.5) 1.0
0.5::a.
b(P) :- subquery(a, P).
query(b(_)). % b(0.5): 1.