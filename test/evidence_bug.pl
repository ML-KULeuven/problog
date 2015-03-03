%Expected outcome:
% a1 0.12
% a2 0.3

0.4::a.
0.3::b.

0.5::q.

a1 :- b, a.
a2 :- b.

query(a1).
query(a2).

evidence(q, true).


