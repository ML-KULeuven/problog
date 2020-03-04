stage(C) :- between(1,2,C).

? :: play(C) :- stage(C).

% ? :: bet(heads).
?::bet(C, tails); ?::bet(C, heads) :- play(C).

0.45::toss(C, heads); 0.55::toss(C, tails) :- stage(C).


win :- play(C),  toss(C, X), bet(C, X).

utility(play(C), -3) :- stage(C).
utility(bet(C, heads), -1) :- stage(C).
utility(bet(C, tails), -1) :- stage(C).
utility(win, 10).