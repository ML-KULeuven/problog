%Expected outcome:
% p 0.85

0.5::a.
0.3::b.

q :- a,b.
p :- \+q.

query(p).