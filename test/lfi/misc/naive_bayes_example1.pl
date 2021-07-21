% Naive bayes example 1
% Source: https://dtai.cs.kuleuven.be/problog/tutorial/learning/03_naivebayes.html
%Expected outcome:
% 0.5::topic(t1).
% 0.5::topic(t2).
% 1.0::word(w1) :- topic(t1).
% 0.0::word(w1) :- \+topic(t1).
% 0.0::word(w2) :- topic(t1).
% 0.0::word(w2) :- \+topic(t1).
% 0.0::word(w1) :- topic(t2).
% 0.0::word(w1) :- \+topic(t2).
% 1.0::word(w2) :- topic(t2).
% 0.0::word(w2) :- \+topic(t2).

t(0.5)::topic(t1).
t(0.5)::topic(t2).

t(_)::word(w1) :- topic(t1).
t(0.0)::word(w1) :- \+topic(t1).
t(_)::word(w2) :- topic(t1).
t(0.0)::word(w2) :- \+topic(t1).

t(_)::word(w1) :- topic(t2).
t(0.0)::word(w1) :- \+topic(t2).
t(_)::word(w2) :- topic(t2).
t(0.0)::word(w2) :- \+topic(t2).