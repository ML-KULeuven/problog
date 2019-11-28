p1~beta(10,30).
p2~beta(10,2).
p3~beta(4,7).

P::fire:- P is p1.
P::alarm:- fire, P is p2.
P::alarm:- \+fire, P is p3.

evidence(alarm).

:-free(p2).
query(density(p1)).
