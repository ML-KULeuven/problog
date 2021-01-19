n_people ~ poisson(6).
more_than_five:- n_people>5.
exactly_five:- n_people=:=5.

query(more_than_five).
query(exactly_five).
