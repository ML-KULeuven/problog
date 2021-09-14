% Naive bayes example 6
% Source: https://dtai.cs.kuleuven.be/problog/tutorial/learning/03_naivebayes.html
%Expected outcome:
% 0.5::topic(t1,D).
% 0.5::topic(t2,D).
% 1.0::word(w1,_,D) :- topic(t1,D).
% 0.0::word(w1,_,D) :- \+topic(t1,D).
% 0.0::word(w2,_,D) :- topic(t1,D).
% 0.0::word(w2,_,D) :- \+topic(t1,D).
% 0.0::word(w1,_,D) :- topic(t2,D).
% 0.0::word(w1,_,D) :- \+topic(t2,D).
% 1.0::word(w2,_,D) :- topic(t2,D).
% 0.0::word(w2,_,D) :- \+topic(t2,D).

t(0.5)::topic(t1,D).
t(0.5)::topic(t2,D).

t(_)::word(w1,_,D)   :- topic(t1,D).
t(0.0)::word(w1,_,D) :- \+topic(t1,D).
t(_)::word(w2,_,D)   :- topic(t1,D).
t(0.0)::word(w2,_,D) :- \+topic(t1,D).

t(_)::word(w1,_,D)   :- topic(t2,D).
t(0.0)::word(w1,_,D) :- \+topic(t2,D).
t(_)::word(w2,_,D)   :- topic(t2,D).
t(0.0)::word(w2,_,D) :- \+topic(t2,D).
