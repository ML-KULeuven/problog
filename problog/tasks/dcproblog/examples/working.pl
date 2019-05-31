normal(20,5)~t.
99/100::cooling.

working:- T as t, T**2<30, cooling.
working:- T as t, T<20.

query(working).
