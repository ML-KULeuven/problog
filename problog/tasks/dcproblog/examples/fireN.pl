% grounding of p2(N) not correct with query_density, need to add between
p1(1)~beta(10,30).
p2(N)~beta(10,2):-between(1,2,N).
p3(N)~beta(4,7):-between(1,2,N).

P::fire:- P is p1(1).
P::alarm:- between(1,2,N),  fire, P is p2(N).
P::alarm:- between(1,2,N), \+fire, P is p3(N).

evidence(alarm).

query_density(p1(1)).
query_density(p2(N)).
