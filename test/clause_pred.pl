%Expected outcome:
% clause(a,b(V_0))  1
% clause(c(d(V_0)),b(V_0))  1

a :- b(X).

query(clause(a,b(Y))).

c(d(X)) :- b(X).

query(clause(c(X),Y)).