%Expected outcome:
% <RAND>::a(W,X,Y,Z); <RAND>::a(Z,X,Y,W) :- W=X, X<Y, Y=Z+1.

t(_)::a(W, X, Y, Z);t(_)::a(Z, X, Y, W) :-W=X,X<Y, Y=Z+1.