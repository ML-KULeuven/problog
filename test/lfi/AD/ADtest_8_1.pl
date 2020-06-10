%Expected outcome:
% 0.4::a(X) ; 0.2::b(Y) ; 0.4::c(Z) :- X=Y, Y=Z.

t(_)::a(X) ; t(_)::b(Y) ; t(_)::c(Z) :- X=Y, Y=Z.