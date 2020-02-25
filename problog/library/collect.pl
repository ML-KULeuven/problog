:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(string)).

collectgroup(CodeBlock, AggVar, GroupBy, Values) :-
    all_or_none((GroupBy, AggVar), CodeBlock, List1),
    enum_groups(List1, GroupBy, Values).

'=>'(CodeBlock, GroupBy/Collector) :-
    Collector =.. [Predicate | Args],
    concat(['collect_', Predicate], NewPredicate),
    CollectorNew =.. [NewPredicate, CodeBlock, GroupBy | Args],
    call(CollectorNew).

'=>'(CodeBlock, Collector) :-
    Collector \= _/_,
    Collector =.. [Predicate | Args],
    concat(['collect_', Predicate], NewPredicate),
    CollectorNew =.. [NewPredicate, CodeBlock, none | Args],
    call(CollectorNew).
    
collect_list(CodeBlock, Y, Result) :-
    collect_list(CodeBlock, none, Y, Result).  % fall back to the grouped version with a dummy group
    
collect_list(CodeBlock, GB, Y, Result) :-
    collectgroup(CodeBlock, Y, GB, Result).