:- table solve(_,lattice(or/3)).

or(P1,P2,(P1;P2)).

solve(true,true) :- !.
solve((A,B),(ProofA,ProofB)) :- !, solve(A,ProofA), solve(B,ProofB).
%solve(A,A:-builtin) :- predicate_property(A, builtin), !, A.
solve(A,A:-ProofB) :- cl(A,B),solve(B,ProofB).

prove(Q,Proofs) :- findall(Proof,solve(Q,Proof),Proofs).