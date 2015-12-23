%System test Negative head literals.
%Description: Decrease the probability of an outcome using negative head literals
%Query: Probability of outcome
%Expected outcome: 
% a 0.315
% d 0.28
% e 0.28
0.7::b.
0.2::c.

0.5::a :- b.
0.5::\+a :- c.

0.5::d :- b.
\+d :- c.

0.5::e :- b.
1.0::\+e :- c.

%%% Queries
query(a).
query(d).
query(e).

