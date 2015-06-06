%Expected outcome:
% ERROR IndirectCallCycleError

a(1).
a(2).

a(X) :- call(a(X)).

query(a(L)).