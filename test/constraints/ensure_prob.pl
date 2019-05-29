%Expected outcome:
% perform_test(1) 1.0
% detect 0.9
% ----------
% perform_test(1) 1.0
% perform_test(2) 1.0
% detect 0.98
% ----------
% perform_test(2) 1.0
% perform_test(3) 1.0
% detect 0.9
% ----------
% perform_test(1) 1.0
% perform_test(3) 1.0
% detect 0.9500000000000001


? :: perform_test(X) :- between(1, 3, X).
0.9 :: detect :- perform_test(1).
0.8 :: detect :- perform_test(2).
0.5 :: detect :- perform_test(3).

ensure_prob(0.90, 1.0) :- detect.

ensure_false :- perform_test(1), perform_test(2), perform_test(3).

query(perform_test(_)).
query(detect).