:- table solve/1.

solve(true) :- !.
solve((A,B)) :- !, solve(A), solve(B).
%solve(A,A:-builtin) :- predicate_property(A, builtin), !, A.
solve(call(A)) :- solve(A),recordz(proof,call(A):-A).
solve(A) :- cl(A,B),solve(B),recordz(proof,A:-B).
solve(A) :- fa(I,P,A),recordz(proof,::(I,P,A)).


prove(Q,Proofs,GroundQueries) :-
    abolish_all_tables,
    forall(recorded(proof,_,Ref),erase(Ref)),
    findall(Q,solve(Q),GroundQueries),
%   solve(Q),
    findall(P,recorded(proof,P),Proofs).
