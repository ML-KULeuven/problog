%Expected outcome:
%                                                    p1(1,1) 1
%                                                    p1(2,2) 1
%                                                    p2(1,1) 1
%                                                    p2(2,2) 1
%                                                    p3(1,1) 1
%                                                    p3(2,2) 1
%                                                  p4(1,1,1) 1
%                                                    p5(1,1) 1
%                                                    p5(2,2) 1
%                                                  p6(1,1,1) 1
%                                                    p7(2,2) 1
%                                       p8([(1, 1), (2, 2)]) 1
%        p9((X4, X5),(a(X4), a(X5), X4=X5),[(1, 1), (2, 2)]) 1
%                                             p10(a(a,b(a))) 1

a(1). a(2).

% Explicit unification.
p1(X,Y) :- X=Y, a(X).
query(p1(_,_)).

% Implicit unification.
eq(X,X).
p2(X,Y) :- eq(X,Y), a(X).
query(p2(_,_)).

% Unification on variable from different scope.
a(X,Y) :- a(X), a(Y).
p3(X,Y) :- X=Y, a(X,Y).
query(p3(_,_)).

% Triplet.
p4(X,Y,Z) :- eq(X,Y), eq(Y,Z), X=1.
query(p4(_,_,_)).

% Unification in query.
p5(X,Y) :- a(X), a(Y).
query(p5(X,X)).


% Triplet with unification in query.
p6(X,Y,Z) :- eq(X,Y), X=1.
query(p6(_,A,A)).

% Unification of variables internal in scope.
b(X,Y) :- a(X).
p7(X,Y) :- b(X,Z1), b(Y,Z2), Z1=Z2, Z1 is X+Y, Z2=4.
query(p7(X,Y)).

% Unification of local variables in findall.
p8(L) :- findall((X,Y), (X=Y,a(X),a(Y)), L).
query(p8(L)).

% Unification of local variables in findall with external pattern.
p9(P,G,L) :- findall(P, G, L).
query(p9((X,Y),(a(X),a(Y),X=Y),L)).

p10(a(X, b(X))).
query(p10(a(a, b(Y)))).


