%Expected outcome:
% 0.449490::a; 0.550510::b.
% 0.449490::a; 0.550510::c.

t(_)::a; t(_)::b.
t(_)::a; t(_)::c.

% ------------BaseProgram------------------------
% a :- lfi_body(0,t)
% lfi_body(0,t) :- lfi_par(0,t), lfi_fact(0,t)
% lfi_par(0,t) :- true
% b :- lfi_body(1,t)
% lfi_body(1,t) :- lfi_par(1,t), lfi_fact(1,t)
% lfi_par(1,t) :- true
%
% a :- lfi_body(2,t)
% lfi_body(2,t) :- lfi_par(2,t), lfi_fact(2,t)
% lfi_par(2,t) :- true
% c :- lfi_body(3,t)
% lfi_body(3,t) :- lfi_par(3,t), lfi_fact(3,t)
% lfi_par(3,t) :- true

% -----------Ground with evidence a:T, b:F, c:F-------------------------
% True::a. # since a:T
% lfi(0,t)::lfi_body(0,t). # lfi_body(0,t) should have been set to true
% lfi(1,t)::b.
% lfi_fact(1,t) :- fail. # since b:F
% lfi_body(1,t) :- fail.
% lfi_par(0,t).
% lfi_par(1,t).
%
% lfi(2,t)::lfi_body(2,t). # lfi_body(0,t) should have been set to true
% lfi(3,t)::c.
% lfi_fact(3,t) :- fail. # since c:F
% lfi_body(3,t) :- fail.
% lfi_par(2,t).
% lfi_par(3,t).
%
% evidence(a).
% evidence(b).
% evidence(c).
%
% query(lfi_fact(2,t)).
% query(lfi_body(2,t)). # why random number?
% query(lfi_fact(0,t)).
% query(lfi_body(0,t)). # why random number?
% query(lfi_fact(3,t)).
% query(lfi_fact(1,t)).
% query(lfi_body(3,t)).
% query(lfi_body(1,t)).
% query(lfi_par(0,t)).
% query(lfi_par(2,t)).
% query(lfi_par(3,t)).
% query(lfi_par(1,t)).

% -----------....I think it should be a:T, b:F, c:F....-------------------------
% lfi(0,t)::a.
% lfi(1,t)::b.

% lfi_body(0,t).

% lfi_fact(1,t) :- fail. # since b:F
% lfi_body(1,t) :- fail.
% lfi_par(0,t).
% lfi_par(1,t).