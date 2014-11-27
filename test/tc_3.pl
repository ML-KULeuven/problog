0.5::stressed(X) :- student(X).
0.2::stressed(X) :- athlet(X).

athlet(1).
athlet(2).
student(2).
student(3).

query(stressed(1)).
query(stressed(2)).
query(stressed(3)).

