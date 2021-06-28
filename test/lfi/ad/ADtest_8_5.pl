%Expected outcome:
% 0.5::a(X,Y) ; 0.125::b(Y,Z) ; 0.375::c(Z,X) :- X=Y, Y=Z.

t(_)::a(X,Y) ; t(_)::b(Y,Z) ; t(_)::c(Z,X) :- X=Y, Y=Z.