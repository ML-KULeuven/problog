%Expected outcome:
% scope1:profit 0.3456

0.4::scope1:cell(1,1,2).
0.4::scope1:cell(1,1,1).

% Unifies for V = 1 and 2
% Two disconnected AD will be created, hence the different proba between profit and cell(1,1,6)
scope1:(0.4::cell(1,1,3);0.6::cell(1,1,6):-cell(1,1,V), V<3).
scope1:(profit :- cell(1,1,X), X > 4, \+ cell(1,1,3)).

query(scope1:profit).
