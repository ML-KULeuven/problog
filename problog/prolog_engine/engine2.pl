:- table solve/1.
:- dynamic cl/2.
:- dynamic fa/3.


solve(true).% :- !.
solve((A,B)) :-writeln(and(A,B)), solve(A), solve(B),writeln(proven_and(A,B)).
solve(call(A)) :-  solve(A),recordz(proof,call(A):-A).
%solve(neg(A)) :- !,\+solve(A).
%solve(neg(A)) :- writeln(neg(A)),findall(A,solve(A),Proofs),writeln(neg(A)=Proofs),Term=..[','|Proofs],recordz(proof,neg(A):-neg(Term)).% length(Proofs,0).
solve(neg(A)) :- copy_term(A,A2), forall(solve(A),recordz(proof,neg(A2):-A)).
solve(A) :- functor(A,F,_), builtin(F),writeln(F=builtin),  A, recordz(proof,A:-builtin(A)).
%solve(A) :- predicate_property(A, foreign), !, A, recordz(proof,A:-foreign(A)).
%solve(A) :- findall(B,(cl(A,B),solve(B)),Proofs),forall(member(B,Proofs),recordz(proof,A:-B)).
solve(A) :- fa(I,P,A),recordz(proof,::(I,P,A)).
solve(A) :- cl(A,B),solve(B),recordz(proof,A:-B).

builtin(>).


prove(Q,Proofs,GroundQueries) :-
    abolish_all_tables,
    forall(recorded(proof,_,Ref),erase(Ref)),
    findall(Q,solve(Q),GroundQueries),
    findall(P,(recorded(proof,P),numbervars(P,0,_)),Proofs).
