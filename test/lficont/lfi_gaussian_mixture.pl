t(0.5)::c.

t(normal(1,10))::fa :- c.
t(normal(10,10))::fa :- \+c.
query(c).
query(fa).

