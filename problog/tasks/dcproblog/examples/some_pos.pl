normal(0,2)~a(C).

pos(C) :- AC as a(C), AC>0.
some_pos(N) :- between(1,N,C), pos(C).

query(some_pos(6)).
