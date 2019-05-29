density(Q):- possible(Q), density_builtin(Q).
Val as Var:- possible(Var), as_builtin(Val, Var).

observation_builtin(Var,Obs):- Val as Var, obs_builtin(Val,Obs).
