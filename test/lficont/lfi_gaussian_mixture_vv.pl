t(0.5)::c(ID,1); t(0.5)::c(ID,2).
comp(1). comp(2).

t(normal(_,_),C)::fa(ID) :- comp(C), c(ID,C).

