t(0.5)::c(ID).

t(normal(1,10))::fa(ID) :- c(ID).
t(normal(10,10))::fa(ID) :- \+c(ID).

