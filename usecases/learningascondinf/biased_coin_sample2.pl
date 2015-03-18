
0.9::coin(h,T) ; 0.1::coin(t,T).

0.2::stop(T).

tosses(C) :- tosses(C,0).
tosses([],T) :- stop(T).
tosses([H|R],T) :- \+stop(T), coin(H,T), Tn is T+1, tosses(R,Tn).

query(tosses(C)).

