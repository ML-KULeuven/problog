machine(1).
machine(2).

temperature ~ normal(20,5).

0.99::cooling(1).
0.95::cooling(2).

works(N):- machine(N), cooling(N).
works(N):- machine(N), temperature<25.0.

evidence(works(2)).
query(works(1)).
