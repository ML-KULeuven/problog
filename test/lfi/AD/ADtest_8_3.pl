%Expected outcome:
% 0.8::a(X,X) ; 0.2::b(Y,Y) ; 0::c(Z) :- X=Y, Y=Z.

t(_)::a(X,X) ; t(_)::b(Y,Y) ; t(_)::c(Z) :- X=Y, Y=Z.