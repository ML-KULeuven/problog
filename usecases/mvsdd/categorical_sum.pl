% Sum of N dice with K faces each, as a showcase for MV-SDD.
%
%   problog categorical_sum.pl -a N -a K -k mvsdd
%   problog categorical_sum.pl -a N -a K -k sdd
%
% Every die is an annotated disjunction over its K faces (K = 2, 4, 8, 16 or 32), and the
% queries ask for the distribution of the total.  MV-SDD represents each die as one
% variable with K values.  SDD gives every face its own Boolean variable, and until the
% exactly-one constraints are added at the end, compiles sum/2 as if several faces of a die
% could be up at once: its intermediate diagrams track sets of reachable sums, which grow
% exponentially with K.  run.py sweeps N and K and times both.

:- use_module(library(lists)).

arg(Position, Value) :- cmd_args(Args), nth1(Position, Args, Atom), atom_number(Atom, Value).
dice(I) :- arg(1, N), between(1, N, I).
faces(K) :- arg(2, K).

1/2::die(2, I, 0); 1/2::die(2, I, 1) :- dice(I).

1/4::die(4, I, 0); 1/4::die(4, I, 1); 1/4::die(4, I, 2); 1/4::die(4, I, 3) :- dice(I).

1/8::die(8, I, 0); 1/8::die(8, I, 1); 1/8::die(8, I, 2); 1/8::die(8, I, 3);
    1/8::die(8, I, 4); 1/8::die(8, I, 5); 1/8::die(8, I, 6); 1/8::die(8, I, 7)
    :- dice(I).

1/16::die(16, I, 0); 1/16::die(16, I, 1); 1/16::die(16, I, 2); 1/16::die(16, I, 3);
    1/16::die(16, I, 4); 1/16::die(16, I, 5); 1/16::die(16, I, 6); 1/16::die(16, I, 7);
    1/16::die(16, I, 8); 1/16::die(16, I, 9); 1/16::die(16, I, 10); 1/16::die(16, I, 11);
    1/16::die(16, I, 12); 1/16::die(16, I, 13); 1/16::die(16, I, 14); 1/16::die(16, I, 15)
    :- dice(I).

1/32::die(32, I, 0); 1/32::die(32, I, 1); 1/32::die(32, I, 2); 1/32::die(32, I, 3);
    1/32::die(32, I, 4); 1/32::die(32, I, 5); 1/32::die(32, I, 6); 1/32::die(32, I, 7);
    1/32::die(32, I, 8); 1/32::die(32, I, 9); 1/32::die(32, I, 10); 1/32::die(32, I, 11);
    1/32::die(32, I, 12); 1/32::die(32, I, 13); 1/32::die(32, I, 14); 1/32::die(32, I, 15);
    1/32::die(32, I, 16); 1/32::die(32, I, 17); 1/32::die(32, I, 18); 1/32::die(32, I, 19);
    1/32::die(32, I, 20); 1/32::die(32, I, 21); 1/32::die(32, I, 22); 1/32::die(32, I, 23);
    1/32::die(32, I, 24); 1/32::die(32, I, 25); 1/32::die(32, I, 26); 1/32::die(32, I, 27);
    1/32::die(32, I, 28); 1/32::die(32, I, 29); 1/32::die(32, I, 30); 1/32::die(32, I, 31)
    :- dice(I).

roll(I, Face) :- faces(K), die(K, I, Face).

sum(0, 0).
sum(I, S) :- I > 0, I1 is I - 1, sum(I1, S1), roll(I, Face), S is S1 + Face.

total(S) :- arg(1, N), sum(N, S).
query(total(_)).
