0.2::hot.
0.01::no_cool.
t~normal(20,5):- \+hot.
t~normal(27,5):- hot.
broken:- t>30.
broken:- no_cool, t>20.

query(broken).
