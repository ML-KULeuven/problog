:- use_module(library(lists)).
:- use_module(library(apply)).
:- use_module(library(string)).

collectgroup(CodeBlock, AggVar, GroupBy, Values) :-
    all_or_none((GroupBy, AggVar), CodeBlock, List1),
    enum_groups(List1, GroupBy, Values).

'=>'(CodeBlock, GroupBy/Collector) :-
    Collector =.. [Predicate | Args],
    concat(['collect_', Predicate], NewPredicate),
    CollectorNew =.. [NewPredicate, CodeBlock | Args],
    call(CollectorNew, GroupBy).

'=>'(CodeBlock, Collector) :-
    Collector \= _/_,
    Collector =.. [Predicate | Args],
    concat(['collect_', Predicate], NewPredicate),
    CollectorNew =.. [NewPredicate, CodeBlock | Args],
    call(CollectorNew, none).
    
collect_list(CodeBlock, Y, Result) :-
    collect_list(CodeBlock, Y, Result, g).  % fall back to the grouped version with a dummy group
    
collect_list(CodeBlock, Y, Result, GB) :-
    collectgroup(CodeBlock, Y, GB, Result).