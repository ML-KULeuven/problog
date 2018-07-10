%Expected outcome:
%    f([1, 2])  0.12
% f([X4, X5, X6])  0
%    f([a, b])  0
% f([c, b, a])  0


%% Your program
0.3::a(1).
0.4::a(2).
f(L) :- findall(A, a(A), L).

query( f([c,b,a]) ).
query( f([X,Y,Z]) ).
query( f([a,b]) ).
query( f([X,Y]) ).

