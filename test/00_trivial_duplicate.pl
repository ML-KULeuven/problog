% Duplicate fact. Interpret as two separate facts.
%Expected outcome:
% p(1) 0.72
% p(2) 0.2

0.3::p(1).
0.2::p(2).
0.6::p(1).

query(p(1)).
query(p(2)).
