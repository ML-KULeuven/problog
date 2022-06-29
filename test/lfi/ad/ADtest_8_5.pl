%Expected outcome:
% 0.6::a(X,Y) ; 0.2::b(Y,Z) ; 0.2::c(Z,X) :- X=Y, Y=Z.

t(_)::a(X,Y) ; t(_)::b(Y,Z) ; t(_)::c(Z,X) :- X=Y, Y=Z.