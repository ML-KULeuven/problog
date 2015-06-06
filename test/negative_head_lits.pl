%System test Negative head literals.
%Description: Decrease the probability of an outcome using negative head literals
%Query: Probability of outcome
%Expected outcome: 
% a 0.315
0.7::b.
0.2::c.

0.5::a :- b.
0.5::\+a :- c.

%%% Queries
query(a).

