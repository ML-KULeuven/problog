%Expected outcome:
% some_pos(3) 7/8
ac(C)~normal(0,2).
pos(C) :- AC is ac(C), AC>0.
some_pos(N) :- between(1,N,C), pos(C).
query(some_pos(3)).
