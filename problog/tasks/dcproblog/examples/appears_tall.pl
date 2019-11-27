2/5::male;3/5::female.

height(1)~normal(160,8):-female.
height(1)~normal(180,8):-male.

is_tall:-male, N is 1, H is height(N), A is H+10, 3>2, 190=<A.
is_tall:-female, N is 1, 170=<(height(N)+10)*0.99, V = {1,2,3}, writeln(V).


appears_tall:- is_tall.
3/10::appears_tall:-male.

query(is_tall).
query(appears_tall).
