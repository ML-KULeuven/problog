%Expected outcome:
% a2 0.12
% a1 0.3

0.4::a.
0.3::b.

0.5::q.

a2 :- b, a.
a1 :- b.

query(a2).
query(a1).

evidence(q, true).


