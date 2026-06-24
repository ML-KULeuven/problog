% Tests for assertz/1, asserta/1 and dynamic/1 declarations
:- use_module(library(assert)).

% Test 1: Basic assertz - fact is added and can be queried
:- assertz(base_fact(1)).
:- assertz(base_fact(2)).
:- assertz(base_fact(3)).

assert_t001(X) :- base_fact(X).
query(assert_t001(X)).
%Expected outcome:
% assert_t001(1) 1
% assert_t001(2) 1
% assert_t001(3) 1

% Test 2: assertz in body - adds facts during evaluation
assertz_setup :-
    assertz(dyn_fact(a)),
    assertz(dyn_fact(b)),
    fail.
assertz_setup.

assert_t002(X) :- assertz_setup, dyn_fact(X).
query(assert_t002(X)).
%Expected outcome:
% assert_t002(a) 1
% assert_t002(b) 1

% Test 3: dynamic/1 allows empty predicate to fail gracefully
:- dynamic(possibly_empty/0).

assert_t003 :- possibly_empty.
query(assert_t003). % outcome: 0

% Test 4: dynamic/1 + assertz works correctly when violation occurs
:- dynamic(compensated/0).

base_cond(a).
base_cond(b).

cond_met :- base_cond(a), base_cond(b).

materialize_compensation :-
    cond_met,
    assertz(compensated),
    fail.
materialize_compensation.

assert_t004 :- materialize_compensation, compensated.
query(assert_t004). % outcome: 1

% Test 5: dynamic/1 + assertz - no compensation when condition not met
:- dynamic(compensated2/0).

base_cond2(x).
% base_cond2(y) is intentionally absent

cond_met2 :- base_cond2(x), base_cond2(y).

materialize_compensation2 :-
    cond_met2,
    assertz(compensated2),
    fail.
materialize_compensation2.

assert_t005 :- materialize_compensation2, compensated2.
query(assert_t005). % outcome: 0

% Test 6: dynamic/1 with multiple predicates
:- dynamic((dyn_multi_a/1, dyn_multi_b/2)).

assert_t006 :- dyn_multi_a(_).
query(assert_t006). % outcome: 0

assert_t007 :- dyn_multi_b(_, _).
query(assert_t007). % outcome: 0

% Test 7: asserta/1 adds a fact (treated same as assertz in ProbLog)
:- asserta(alist_fact(z)).
:- asserta(alist_fact(y)).

assert_t008(X) :- alist_fact(X).
query(assert_t008(X)).
%Expected outcome:
% assert_t008(y) 1
% assert_t008(z) 1
