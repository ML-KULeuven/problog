%Expected outcome:
% <RAND>::a(X); <RAND>::b(Y) :- X==Y, between(1,2,X), between(1,2,Y).

t(_)::a(X); t(_)::b(Y) :- X==Y, between(1,2,X), between(1,2,Y).

