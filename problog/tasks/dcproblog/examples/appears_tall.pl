2/5::male;3/5::female.

normal(160,8)~height(1):-female.
normal(180,8)~height(1):-male.

is_tall:-male, N is 1, H as height(N), A is H+10, 3>2, 190=<A.
is_tall:-female, H as height(1), A is 10, B is A+H, 170=<B*0.99.

appears_tall:- is_tall.
3/10::appears_tall:-male.

query(is_tall).
query(appears_tall).
