beta(1,1)~b.
B::coin_flip(N):- B as b.

evidence(coin_flip(1), true).
evidence(coin_flip(2), false).

:-free(b).
query(density(b)).
