% Noisy-or LFI example 4
% Source: https://dtai.cs.kuleuven.be/problog/tutorial/learning/04_noisyor.html
%Expected outcome:
% counts(C) :- between(0,5,C).
% 0.0001::word(_,_).
% 0.0::topic(t1).
% 0.666629622634077::topic(t1) :- counts(C), word(w1,C).
% 0.0::topic(t1) :- counts(C), word(w2,C).
% 0.0::topic(t2).
% 0.0::topic(t2) :- counts(C), word(w1,C).
% 1.0::topic(t2) :- counts(C), word(w2,C).

counts(C) :- between(0,5,C).
0.0001::word(_,_).

t(_)::topic(t1). % leak probability
t(_)::topic(t1) :- counts(C), word(w1,C).
t(_)::topic(t1) :- counts(C), word(w2,C).

t(_)::topic(t2). % leak probability
t(_)::topic(t2) :- counts(C), word(w1,C).
t(_)::topic(t2) :- counts(C), word(w2,C).
