0.3::p(1); 0.4::p(2) <- true.

all :- p(1), p(2).

none :- \+p(1), \+p(2).

any :- p(1); p(2).

% evidence(none, true).

query(p(1)).
query(p(2)).
query(all).
query(none).
query(any).