? :: play.

% ? :: bet(heads).
?::bet(tails); ?::bet(heads) :- play.

0.45::toss(heads); 0.55::toss(tails).


win :- play,  toss(X), bet(X).

utility(play, -5).
utility(bet(heads), -1).
utility(bet(tails), -1).
utility(win, 10).

