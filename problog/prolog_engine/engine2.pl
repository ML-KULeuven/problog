:- table solve/1.
:- dynamic cl/2.
:- dynamic fa/3.

solve(true) :- !.
solve((A,B)) :- !, solve(A), solve(B).
solve(call(A)) :- !, solve(A),recordz(proof,call(A):-A).
solve(neg(A)) :- !,\+solve(A).
solve(A) :- builtin(A), !, A, recordz(proof,A:-builtin(A)).
%solve(A) :- predicate_property(A, foreign), !, A, recordz(proof,A:-foreign(A)).

builtin(>).

solve(A) :- cl(A,B),solve(B),recordz(proof,A:-B).
solve(A) :- fa(I,P,A),recordz(proof,::(I,P,A)).


prove(Q,Proofs,GroundQueries) :-
    abolish_all_tables,
    forall(recorded(proof,_,Ref),erase(Ref)),
    findall(Q,solve(Q),GroundQueries),
%   solve(Q),
    findall(P,recorded(proof,P),Proofs).
