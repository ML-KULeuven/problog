t~normal(20,5).
99/100::cooling.

working:- T is t, T**2<30, cooling.
working:- T is t, T<20.

query(working).
