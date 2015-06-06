%Expected outcome:
% ERROR CallModeError

%% Your program
0.3::a(1).
0.4::a(2).
f(L) :- findall(A, a(A), L).
query( f(abcdef) ).
query(f(s(r(23),(17,b)))).


