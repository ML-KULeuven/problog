%Expected outcome:
% q1 0.4

sample([X|L],X) :- sample_now([X|L]).
sample([H|L],X) :- \+ sample_now([H|L]), sample(L,X).

P::sample_now(L) :- length(L,N), P is 1.0/N.

q1 :- sample([a,a,b,c,c],c).

query(q1).

