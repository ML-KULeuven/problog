2/5::male;3/5::female.

normal(160,8)~height(1):-female.
normal(180,8)~height(1):-male.

% is_tall:-male, sample_uniform1(3,[1,2],R), N is 1, H as height(N), A is H+10, 3>2, 190=<A.
% is_tall:-female, H as height(1), A is 10, B is A+H, 170=<B*0.99.
% is_tall:-female, A is 10, B is height(1), C as height(1), 170=<C*0.99.


is_tall:- H is height(1), A is 5*2, height(1)<170, 4<5.
% is_tall:- A is 5*2, writeln(A).


appears_tall:- is_tall.
3/10::appears_tall:-male.

query(is_tall).
query(appears_tall).
