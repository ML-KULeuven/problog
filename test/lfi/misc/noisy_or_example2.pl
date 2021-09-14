% Noisy-or LFI example 2
% Source: https://dtai.cs.kuleuven.be/problog/tutorial/learning/04_noisyor.html
%Expected outcome:
% 0.0001::word(_).
% 0.0::topic(t1).
% 0.0::topic(t1) :- word(w1).
% 0.0::topic(t1) :- \+word(w1).
% 0.0::topic(t1) :- word(w2).
% 1.0::topic(t1) :- \+word(w2).
% 0.0::topic(t2).
% 0.0::topic(t2) :- word(w1).
% <RAND>::topic(t2) :- \+word(w1).
% 1.0::topic(t2) :- word(w2).
% 0.0::topic(t2) :- \+word(w2).

0.0001::word(_).

t(_)::topic(t1). % leak probability
t(_)::topic(t1) :-   word(w1).
t(_)::topic(t1) :- \+word(w1).
t(_)::topic(t1) :-   word(w2).
t(_)::topic(t1) :- \+word(w2).

t(_)::topic(t2). % leak probability
t(_)::topic(t2) :-   word(w1).
t(_)::topic(t2) :- \+word(w1).
t(_)::topic(t2) :-   word(w2).
t(_)::topic(t2) :- \+word(w2).

