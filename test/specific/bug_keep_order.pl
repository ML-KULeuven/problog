%Expected outcome:
%  endStateContains([table(a), on(b,a), on(c,b), on(d,c), clear(d)],[while([moveToTable(d,c), moveToTable(c,b), moveToTable(b,a)],8)])        0.8327102464

:- use_module(library(lists)).
moveToTable(A, B, BT, ET, Step) :-
	moveToTable1(Step),
	member(clear(A), BT),
	member(on(A,B), BT),
	Temp0 = [table(A), clear(B) |BT],
	delete(Temp0, on(A,B), ET).

moveToTable(A, B, BT, BT, Step) :-
	moveToTable2(Step),
	member(clear(A), BT),
	member(on(A,B), BT).


0.4::moveToTable1(Step) ; 0.6::moveToTable2(Step).

exec(BT, [], BT, _).
exec(BT, [moveToTable(A, B)|R], ET, StepN) :-
	moveToTable(A, B, BT, TempT, StepN),
	StepNPlusOne is StepN + 1,
	exec(TempT, R, ET, StepNPlusOne).

exec(BT, [moveToTable(A, B)|R], ET, StepN) :-
	\+moveToTable(A, B, BT, TempT, StepN),
	StepNPlusOne is StepN + 1,
	exec(BT, R, ET, StepNPlusOne).

conditionIsTrue(State, falseInState(clear(a))) :-
	\+ member(clear(a), State).


%while
exec(BT, [while(Do, Limit) | R], ET, StepN) :-
	limitedWhile(BT, while(Do, Limit), StepN, TempT),
	flatten([Do], FlattenedDo),
	length(FlattenedDo, DoLength),
	ConsumedSteps is DoLength*Limit,
	NewStepN is StepN + ConsumedSteps,
	exec(TempT, R, ET, NewStepN).

%execWhile
limitedWhile(BT,while(_,0),_,BT).

limitedWhile(BT,while(Do, Limit),_, BT) :- member(clear(a), BT).

limitedWhile(BT, while(Do, Limit), StepN, EndState) :-
	Limit \= 0,
	\+ member(clear(a), BT),
	flatten([Do], NewPlan),
	exec(BT, NewPlan, ET, StepN),
	NewLimit is Limit - 1,
	length(NewPlan, NewPlanLength),
	NewStepN is StepN + NewPlanLength,
	limitedWhile(ET, while(Do, NewLimit), NewStepN, EndState).

endStateContains(BeginState, Plan) :-
	exec(BeginState, Plan, EndState, 1),
	subset([clear(a)], EndState).

query(endStateContains([table(a), on(b,a),on(c,b),on(d,c), clear(d)], [while([moveToTable(d,c), moveToTable(c,b), moveToTable(b,a)], 8)])).