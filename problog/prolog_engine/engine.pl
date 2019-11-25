solve(true,true) :- !.
solve((A,B), (ProofA,ProofB)) :- !, solve(A,ProofA), solve(B,ProofB).
solve(A,A:-builtin) :- predicate_property(A, builtin), !, A.
solve(A,(A:-(Proof,p(P)))) :- cl(A,B,P),solve(B,Proof).

prove(Q,Proofs) :- findall(P,solve(Q,P), Proofs), writeln(Proofs).