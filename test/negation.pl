%Expected outcome:
% q1 0.14
% q2 0.06

0.2::a.
0.7::b.

c :- a,b.
c :- a,\+b.

q1 :- b, c.
q2 :- \+ b, c.


query(q1).
query(q2).

