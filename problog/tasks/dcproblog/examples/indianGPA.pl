1/4::american;3/4::indian.

19/20::isdensity(a):-american.
99/100::isdensity(i):-indian.

17/20::perfect_gpa(a):-american.
1/10::perfect_gpa(i):-indian.


gpa(a)~uniform(0,4):- isdensity(a).
gpa(a)~delta(4.0):-\+isdensity(a), perfect_gpa(a).
gpa(a)~delta(0.0):-\+isdensity(a), \+perfect_gpa(a).

gpa(i)~uniform(0,10):- isdensity(i).
gpa(i)~delta(10.0):-\+isdensity(i), perfect_gpa(i).
gpa(i)~delta(0.0):-\+isdensity(i), \+perfect_gpa(i).

gpa(student)~delta(A):- A is gpa(a).
gpa(student)~delta(I):- I is gpa(i).

observation(gpa(student),2.0).
query(american).
query(indian).
