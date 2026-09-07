%Expected outcome:
% flat 0
% nested 0
% nested_left 0
% nested_right 0
% consistent 0.12

% Regression test for GitHub issue #145: a parenthesized (i.e. left-nested)
% conjunction that is unsatisfiable was reported as if it succeeded.

0.2::a.
0.6::c.

flat :- \+ c, c, a.
nested :- ((\+ c, c), a).
nested_left :- (c, \+ c), a.
nested_right :- a, (\+ c, c).
consistent :- (a, c).

query(flat).
query(nested).
query(nested_left).
query(nested_right).
query(consistent).
