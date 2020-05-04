:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(string)).
:- use_module(library(subqueries)).

collectgroup(CodeBlock, AggVar, GroupBy, Values) :-
    export_findall((GroupBy, AggVar), CodeBlock, List1),
    enum_groups(List1, GroupBy, Values).

'=>'(CodeBlock, CB, GroupBy/Collector) :-
    writeln("HEY").

'=>'(CodeBlock, GroupBy/Collector) :-
    writeln(GroupBy), writeln(Collector), writeln(CodeBlock),
    Collector =.. [Predicate | Args],
    concat(["collect_", Predicate], NewString),
    str2atom(NewString, NewPredicate),
    CollectorNew =.. [NewPredicate, CodeBlock, GroupBy | Args],
    call(CollectorNew).

'=>'(CodeBlock, Collector) :-
    Collector \= _/_,
    Collector =.. [Predicate | Args],
    concat(["collect_", Predicate], NewString),
    str2atom(NewString, NewPredicate),
    CollectorNew =.. [NewPredicate, CodeBlock, none | Args],
    call(CollectorNew).
    
collect_list(CodeBlock, Y, Result) :-
    collect_list(CodeBlock, none, Y, Result).  % fall back to the grouped version with a dummy group
    
collect_list(CodeBlock, GB, Y, Result) :-
    collectgroup(CodeBlock, Y, GB, Result).