%Expected outcome:
% b(0.5) 1.0
% c(0.5) 1.0
0.5::a.
b(P) :- subquery(a, P).
c(P) :- subquery(a, P, [], "logprob", "ddnnf").
query(b(_)). % b(0.5): 1.
query(c(_)). % c(0.5): 1.
