%Expected outcome:
% 0.5::a(X,Y); 0.5::b(X,Y) :- X=Y, between(1,2,X), between(1,2,Y).

t(_)::a(X,Y); t(_)::b(X,Y) :- X=Y, between(1,2,X), between(1,2,Y).