%Expected outcome:
% is_tall 0.44901267200110784

2/5::male;3/5::female.

height(1)~normal(160,8):-female.
height(1)~normal(180,8):-male.

is_tall:-male, N is 1, H is height(N), A is H+10, 3>2, 190=<A.
is_tall:-female, N is 1, 170=<(height(N)+10)*0.99.

query(is_tall).
