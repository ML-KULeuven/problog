/** comment **/
or(X,Y) :- X;Y.
colon(X,Y,R) :- R is X : Y.
power(X,Y,R) :- R is X ** Y.
soft_cut(X,Y,Z) :- X *-> Y; Z.
xor(X,Y) :- X # 2.
multiplication(X,Y,R) :- R is X * Y.
plus(X,Y,Z) :- Z is X + Y.
if_then(X,Y) :- X -> Y.
%%qsdfqsdf(X,Y) :- X --> Y.
.5::pointNumber.
bitwiseAnd(X,Y) :- X /\ Y.
integerDivision(X,Y,R) :- R is X // Y.
bitwiseShift(X,Y,R) :- R is X << Y.
lessOrEqual(X,Y) :- X =< Y.
colonEqual(X,Y) :- X =:= Y.
notEqual(X,Y) :- X =\= Y.
atEqual(X,Y) :- X =@= Y.
equalDots(X,Y) :- X =.. Y.
normalEqual(X,Y) :- X == Y.
greaterGreater(X,Y,R) :- R is X >> Y.
greaterLess(X,Y) :- X >< Y.
atLess(X,Y) :- X @< Y.
atEqualsLess(X,Y) :- X @=< Y.
atGreaterEquals(X,Y) :- X @>= Y.
atGreater(X,Y) :- X @> Y.
backslashBackslash(X) :- \\X.
backslashAtEquals(X,Y) :- X \=@= Y.
backslashSingleEquals(X,Y) :- X \= Y.
slashBackslash(X,Y) :- X \/ Y.
singleBackslash(X) :- \X.
caretPower(X,Y,R) :- R is X ^ Y.
tilde(X) :- ~X.
parse_operator(X) --> a(X).