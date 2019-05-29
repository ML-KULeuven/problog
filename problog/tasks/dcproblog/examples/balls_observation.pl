n(1).
3/10::material(N,wood);7/10::material(N,metal):- n(1).

beta(2,3)~size(N):-material(N,metal).
beta(4,2)~size(N):-material(N,wood).

observation(size(1),4/10).
query(material(1,wood)).
