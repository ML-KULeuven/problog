%Expected outcome:
% scope1:a 0.3
% scope1:b 0.5
% scope1:a(1) 1
% scope1:a(2) 0.9

0.3::scope1:a.
0.5::scope1:b.

scope1:a(1).
0.9::scope1:a(2).

query(scope1:a).
query(scope1:b).
query(scope1:a(_)).
