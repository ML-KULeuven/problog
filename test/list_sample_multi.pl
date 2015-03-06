%Expected outcome:
% q1 0.014705882

sample(L,N,S) :- permutation1(S,T), sample_ordered(L,N,T).

sample_ordered(_, 0, []).
sample_ordered([X|L], N, [X|S]) :- N > 0, sample_now([X|L],N), N2 is N-1, sample_ordered(L,N2,S).
sample_ordered([H|L], N, S) :- N > 0, \+ sample_now([H|L],N), sample_ordered(L,N,S).

P::sample_now(L,N) :- length(L, M), M >= N, P is N/M.

appendlist1([], X, X).
appendlist1([T|H], X, [T|L]) :- appendlist1(H, X, L).

permutation1([], []).
permutation1([T|H], X) :- permutation1(H, H1), appendlist1(L1, L2, H1), appendlist1(L1, [T], X1), appendlist1(X1, L2, X).

q1 :- sample([a,a,b,b,b,c,c,d,d,d,e,e,f,f,t,t,t,z],3,[c,a,t]).

query(q1).

