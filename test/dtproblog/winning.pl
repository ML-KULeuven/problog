0.5::play1.
0.5::play2.


0.3::solo_win1.
0.3::solo_win2.

0.5::both_win :- not solo_win1, not solo_win2.

win1 :- play1, solo_win1.
win1 :- play1, both_win.

win2 :- play2, solo_win2.
win2 :- play2, both_win.

decision(play1).
decision(play2).

utility(play1, -10).
utility(play2, -10).
utility(win1, 50).
utility(win2, 50).




