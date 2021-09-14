% Naive bayes example 3
% Source: https://dtai.cs.kuleuven.be/problog/tutorial/learning/03_naivebayes.html
% The random probabilities are not entirely random, one pair of probabilities will be close to 1.0
%Expected outcome:
% is_topic(t1).
% is_topic(t2).
% is_word(w1).
% is_word(w2).
% 0.5::topic(t1).
% 0.5::topic(t2).
% <RAND>::word(w1) :- is_topic(t1), is_word(w1), topic(t1).
% 0.0::word(w1) :- is_topic(t2), is_word(w1), topic(t2).
% 0.0::word(w2) :- is_topic(t1), is_word(w2), topic(t1).
% <RAND>::word(w2) :- is_topic(t2), is_word(w2), topic(t2).
% <RAND>::word(w1) :- is_topic(t2), is_word(w1), \+topic(t2).
% 0.0::word(w1) :- is_topic(t1), is_word(w1), \+topic(t1).
% <RAND>::word(w2) :- is_topic(t1), is_word(w2), \+topic(t1).
% 0.0::word(w2) :- is_topic(t2), is_word(w2), \+topic(t2).

is_topic(t1). is_topic(t2).
is_word(w1). is_word(w2).

t(_,T)::topic(T).
t(_,W,T)::word(W) :- is_topic(T), is_word(W), topic(T).
t(_,W,T)::word(W) :- is_topic(T), is_word(W), \+topic(T).