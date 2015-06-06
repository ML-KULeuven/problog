%Expected outcome:
% ERROR IndirectCallCycleError

a(1).
a(2).

a(L) :- findall( X, a(X), L).


query(a(L)).