:- module(string, [sub_string/5, concat/2, join/3, str2lst/2, lst2str/2]).

:- use_module(library('string.py')).
:- use_module(library(lists)).

sub_string(String, Before, Length, After, SubString) :-
    str2lst(String, List),
    sub_list(List, BeforeL, Length, AfterL, SubListL),
    join('', BeforeL, Before),
    join('', AfterL, After),
    join('', SubListL, SubString).



    
