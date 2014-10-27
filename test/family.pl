parent(erik,katrien).
parent(katrien,liese).


ancestor(X,Y) :- parent(X,Y).
ancestor(X,Y) :- parent(X,Y),ancestor(Y,Z). 