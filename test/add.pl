%Expected outcome:  
% all 0
% none 0.3
% any 0.7
% p(1) 0.3
% p(2) 0.4

0.3::p(1); 0.4::p(2).

all :- p(1), p(2).

none :- \+p(1), \+p(2).

any :- p(1); p(2).

query(p(1)).
query(p(2)).
query(all).
query(none).
query(any).