%Expected outcome:
% q1 0.14
% q2 0.06

% Possible worlds:
% ab 0.14 => c, q1, not q2 
% a 0.06 => c, not q1, q2
% b 0.56 => not q1, not q2
% {} 0.24 => not q1, not q2

0.2::a.
0.7::b.

c :- a,b.
c :- a,\+b.

q1 :- b, c.
q2 :- \+ b, c.


query(q1).
query(q2).



