beta(2,5)~bias.
B::tossResult(N):-between(1,2,N), bias~=B.

evidence(tossResult(1),true).
evidence(tossResult(2),false).


B::q:- bias~=B.

:-free_list([bias]).
query(q).
