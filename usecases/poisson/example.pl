% Knowledge about time periods.
morning(T) :- T >= 6, T =< 10.
evening(T) :- T >= 16, T =< 19.

% Knowledge about type or areas.
0.8 :: residential(1).
0.9 :: residential(2).

0.5 :: residential(3).
0.5 :: business(3).

0.8 :: business(4).
0.9 :: business(5).

has_airport(4).

% Morning.
poisson(50) :: request(AreaFrom,AreaTo,TimeSlot) :-
   residential(AreaFrom),
   business(AreaTo),
   morning(TimeSlot).

poisson(3) :: request(AreaFrom,AreaTo,TimeSlot) :-
   business(AreaFrom),
   residential(AreaTo),
   morning(TimeSlot).


% Evening
poisson(50) :: request(AreaFrom,AreaTo,TimeSlot) :-
   business(AreaFrom),
   residential(AreaTo),
   evening(TimeSlot).

poisson(3) :: request(AreaFrom,AreaTo,TimeSlot) :-
   residential(AreaFrom),
   business(AreaTo),
   evening(TimeSlot).


% Always some traffic from and to the airport.
poisson(10) :: request(_, AreaTo, TimeSlot) :-
   has_airport(AreaTo),
   TimeSlot >= 5,
   TimeSlot =< 24.

poisson(10) :: request(AreaFrom, _, TimeSlot) :-
   has_airport(AreaFrom),
   TimeSlot >= 5,
   TimeSlot =< 24.

% There always is some requests.
%poisson(1) :: request(_A,_B,_C).

query( request(1,4,12) ).
query( request(3,3,8) ).
query( request(2,4,8) ).
query( request(5,1,14) ).