:- use_module(library('string.py')).
:- use_module(library(lists)).

sub_string(String, Before, Length, After, SubString) :-
    str2lst(String, List),
    sub_list(List, BeforeL, Length, AfterL, SubListL),
    join('', BeforeL, Before),
    join('', AfterL, After),
    join('', SubListL, SubString).



    
