%Expected outcome:
% scope1:a 0.3
% scope1:b 0.5
% scope1:a 0.3
% scope1:b 0.5

0.3::scope1:a.
0.5::scope1:b.

query(scope1:_).
query(_:a).
query(_:b).
