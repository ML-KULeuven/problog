:- table solve/1.
:- dynamic cl/2.
:- dynamic fa/3.


solve(true).% :- !.
solve((A,B)) :- solve(A), solve(B).
solve(call(A)) :-  solve(A),recordz(proof,call(A):-A).
solve(neg(A)) :- copy_term(A,A2), forall(solve(A),recordz(proof,neg(A2):-A)).
solve(A) :- functor(A,F,_),builtin(F),A, recordz(proof,A:-builtin(A)).
solve(A) :- predicate_property(A, foreign), !, A, recordz(proof,A:-foreign(A)).
solve(A) :- fa(I,P,A),recordz(proof,::(I,P,A)).
solve(A) :- cl(A,B),solve(B),recordz(proof,A:-B).

builtin(>).
builtin(is).
builtin(between).
builtin(member).
builtin(<).
builtin(succ).

prove(Q,Proofs,GroundQueries) :-
    abolish_all_tables,
    forall(recorded(proof,_,Ref),erase(Ref)),
    findall(Q,(solve(Q),numbervars(Q,0,_)),GroundQueries),
    findall(P,(recorded(proof,P),numbervars(P,0,_)),Proofs).
