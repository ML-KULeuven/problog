%Expected outcome:
% a 0.33973214285714293
:-use_module(library(lists)).

l(L):- L = [1,2,3,4,5,6,7,8].
n(X):- l(L), select_uniform(nballs, L, X, _).

3/10::material(N,wood);7/10::material(N,metal):- n(N).

1/3::color(N,grey);1/3::color(N,blue);1/3::color(N,black):- material(N,metal).
1/2::color(N,black);1/2::color(N,brown):- material(N,wood).


size(N)~beta(2,3):-material(N,metal).
size(N)~beta(4,2):-material(N,wood).

dl(Urn,L):- n(N), findall(M, between(1,N,M), L).
drawn(Urn,X):- dl(Urn,L), select_uniform((Urn,L), L, X, _).

a:- drawn(1,A), drawn(2,A).
query(a).
