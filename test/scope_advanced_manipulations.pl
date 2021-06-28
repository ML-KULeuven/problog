%Expected outcome:
% scope2:cell(1,1,3) 0.4
% scope2:cell(1,1,6) 0.6
% scope2:profit 0.6


% We no longer support this syntax (scoped clause).
% scope1:(0.4::cell(1,1,3);0.6::cell(1,1,6):-true).
0.4::scope1:cell(1,1,3); 0.6::scope1:cell(1,1,6).

scope2:cell(1,1,X) :- scope1:cell(1,1,X).

% We no longer support this syntax (scoped clause).
% scope2:(profit :- cell(1,1,X), X > 4, \+ cell(1,1,3)).
scope2:profit :- scope2:cell(1,1,X), X > 4, \+ scope2:cell(1,1,3).

query(scope2:cell(1,1,_)).
query(scope2:profit).