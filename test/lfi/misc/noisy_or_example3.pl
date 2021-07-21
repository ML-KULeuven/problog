% Noisy-or LFI example 3
% Source: https://dtai.cs.kuleuven.be/problog/tutorial/learning/04_noisyor.html
%Expected outcome:
% 0.0::topic(t1).
% 0.666644441481431::topic(t1) :- word(w1,_).
% 0.0::topic(t1) :- word(w2,_).
% 0.0::topic(t2).
% 0.0::topic(t2) :- word(w1,_).
% 1.0::topic(t2) :- word(w2,_).
% 0.0001::word(w1,0).
% 0.0001::word(w1,1).
% 0.0001::word(w1,2).
% 0.0001::word(w1,4).
% 0.0001::word(w2,0).
% 0.0001::word(w2,4).
% 0.0001::word(w2,5).

t(_)::topic(t1). % leak probability
t(_)::topic(t1) :- word(w1,_).
t(_)::topic(t1) :- word(w2,_).

t(_)::topic(t2). % leak probability
t(_)::topic(t2) :- word(w1,_).
t(_)::topic(t2) :- word(w2,_).

% Leak probabilities
0.0001::word(w1,0).
0.0001::word(w1,1).
0.0001::word(w1,2).
0.0001::word(w1,4).
0.0001::word(w2,0).
0.0001::word(w2,4).
0.0001::word(w2,5).