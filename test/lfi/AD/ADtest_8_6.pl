%Expected outcome:
% <RAND>::a(W,X,Y,Z); <RAND>::b(W, X, Y, Z) :-W=X,X<Y, Y=Z+1.

t(_)::a(W, X, Y, Z);t(_)::b(W, X, Y, Z) :-W=X,X<Y, Y=Z+1.