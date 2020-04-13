n(1).
3/10::material(N,wood);7/10::material(N,metal):- n(N).

size(N)~beta(2,3):-material(N,metal).
size(N)~beta(4,2):-material(N,wood).

observation(size(N),P):- P is 4/10, writeln(P).
query(material(1,wood)).
