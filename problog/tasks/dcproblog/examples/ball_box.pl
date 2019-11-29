b~beta(2,2).

P::p:- P is b.

box(1):-p.
box(2):- \+p.

1/4::ball(X, red);3/4::ball(X, white):- box(1).
3/4::ball(X, red);1/4::ball(X, white):- box(2).


evidence(ball(1,red)).
evidence(ball(2,red)).

:-free(b).
query(density(b)).
