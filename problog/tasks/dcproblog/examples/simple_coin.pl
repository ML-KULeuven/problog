b~beta(1,1).
B::coin_flip(N):- B is b.

evidence(coin_flip(1), true).
evidence(coin_flip(2), false).

query_density(b).
