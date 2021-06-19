%Expected outcome:
% 0.5::a(X,X) ; 0.25::b(Y,Y) ; 0.25::c(Z) :- X=Y, Y=Z.

t(_)::a(X,X) ; t(_)::b(Y,Y) ; t(_)::c(Z) :- X=Y, Y=Z.