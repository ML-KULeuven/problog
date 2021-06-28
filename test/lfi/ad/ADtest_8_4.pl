%Expected outcome:
% 0.5::a(W,X) ; 0.125::b(X,Y) ; 0.375::c(Y,Z) :- W=X, X=Y, Y=Z.

t(_)::a(W,X) ; t(_)::b(X,Y) ; t(_)::c(Y,Z) :- W=X, X=Y, Y=Z.