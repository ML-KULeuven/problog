0.5::male(P); 0.5::female(P).

normal(172,30)::height(P) :- male(P).
normal(168,30)::height(P) :- female(P).

hits_head(Person,Height) :- sample(height(Person),H), H >= Height.
cant_see(Person,Height) :- sample(height(Person),H), H =< Height.

query(height(p1)).
query(hits_head(p1,190)).
query(cant_see(p1,160)).