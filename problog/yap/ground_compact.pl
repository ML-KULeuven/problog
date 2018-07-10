% Load original grounder
:- consult(ground_base).

% Load modification (write_fact, write_clause)
:- reconsult(ground_base_compact).

:- (main -> true; halt(12)).