0.3::c.
0.2::a:-c.
b~beta(1,1):-a.
b~beta(2,2):-\+a.
% B::coin_flip(N):- B is b.
%
% evidence(coin_flip(1), true).
% evidence(coin_flip(2), false).
observation(b,0.4).
query_density(b, [b]).
