% From Declarative Data Generation with ProbLog, A. Dries
normal(180,8) :: height(harry).
fixed(H2) :: twice_height(P) :-
	value(height(P),H),
	H2 is H*2.

query(twice_height(X)).