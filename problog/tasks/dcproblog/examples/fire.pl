beta(10,30)~p1.
beta(10,2)~p2.
beta(4,7)~p3.

P::fire:- p1~=P.
P::alarm:- fire, p2~=P.
P::alarm:- \+fire, p3~=P.

evidence(alarm).

:-free(p2).
query(density(p1)).
