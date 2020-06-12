:- table solve/1.
:- dynamic cl/2.
:- dynamic fa/3.


solve(true) :- !.
solve((A,B)) :- solve(A), solve(B).
solve((A;B)) :- solve(A); solve(B).
solve(call(A)) :-  solve(A),recordz(proof,call(A):-A).
%% Multiple args call
% solve(A) :- A =.. [call, Func | Args2], Func =.. [Functor | Args1], append(Args1, Args2, Args),
%            Term =.. [Functor | Args], solve(Term), recordz(proof,A:-Term).
solve(neg(A)) :- copy_term(A,A2), forall(solve(A),(A2 \== A, recordz(proof,A2:-A);true)).
solve(A) :- functor(A,F,_),builtin(F), A, recordz(proof,A:-builtin(A)).
solve(A) :- predicate_property(A, foreign), !, A, recordz(proof,A:-foreign(A)).
solve(A) :- fa(I,P,A), recordz(proof,::(I,P,A)).
solve(A) :- cl(A,B), solve(B),recordz(proof,A:-B).

builtin(>).
builtin(is).
builtin(between).
builtin(member).
builtin(<).
builtin(succ).
builtin(length).

prove(Q,Proofs,GroundQueries) :-
    abolish_all_tables,
    forall(recorded(proof,_,Ref),erase(Ref)),
    findall(Q,(solve(Q),numbervars(Q,0,_)),GroundQueries),
    findall(P,(recorded(proof,P),numbervars(P,0,_)),Proofs).
