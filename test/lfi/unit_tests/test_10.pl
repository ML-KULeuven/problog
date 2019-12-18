%Expected outcome:
% 0.75::a; 0.25::b.

t(_)::a; t(_)::b.

% -----------Ground with evidence a:T, b:F-------------------------
% lfi(0,t)::a.
% lfi(1,t)::b.
% lfi_fact(0,t).
% lfi_body(0,t).
% lfi_par(0,t).
% lfi_par(1,t).
% lfi_fact(1,t) :- fail.
% lfi_body(1,t) :- fail.
%
% evidence(a).
% evidence(b).
%
% query(lfi_fact(0,t)).
% query(lfi_body(0,t)).
% query(lfi_par(0,t)).
% query(lfi_par(1,t)).
% query(lfi_fact(1,t)).
% query(lfi_body(1,t)).