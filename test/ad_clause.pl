%Expected outcome:
% qH 0
% qT 0

biddingH(i1,1,[a1,a2,a3]).
1/3::getsH(A1,I,T); 1/3::getsH(A2,I,T); 1/3::getsH(A3,I,T) <- biddingH(I,T,[A1,A2,A3]).

qH :- getsH(a1,I,T), getsH(a2,I,T).
query(qH).

biddingT(I,T,L) :- I = i1, T = 1, L = [a1,a2,a3].
1/3::getsT(A1,I,T); 1/3::getsT(A2,I,T); 1/3::getsT(A3,I,T) <- biddingT(I,T,[A1,A2,A3]).

qT :- getsT(a1,I,T), getsT(a2,I,T).
query(qT).
