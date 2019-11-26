:- table solve/1.

solve(true) :- !.
solve((A,B)) :- !, solve(A), solve(B).
%solve(A,A:-builtin) :- predicate_property(A, builtin), !, A.
solve(A) :- cl(A,B),solve(B),recordz(proof,A:-B).

prove(Q,Proofs) :-
    abolish_all_tables,
    forall(recorded(proof,_,Ref),erase(Ref)),
    solve(Q),
    findall(P,recorded(proof,P),Proofs).

cl(a(1),true).
cl(b(1),true).
cl(a(2),true).
cl(b(2),true).
cl(c(X),(a(X),b(X))).
cl(b(X),c(X)).
