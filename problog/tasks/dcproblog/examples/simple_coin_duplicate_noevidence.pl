1/5::a.
b~beta(1,1):-a.
b~beta(1,2):-\+a.
B::coin_flip(N):- B is b.

query_density(b).
