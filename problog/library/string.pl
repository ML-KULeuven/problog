:- use_module(library('string.py')).


sub_string(String, Before, Length, After, SubString) :-
    str2lst(String, List),
    sub_list(List, BeforeL, Length, AfterL, SubListL),
    join('', BeforeL, Before),
    join('', AfterL, After),
    join('', SubListL, SubString).

sub_list(List, Before, Length, After, SubList) :-
    sub_list(List, Before, 0, Length, After, SubList).
    
sub_list([X|List], [X|Before], 0, Length, After, SubList) :-
    sub_list(List, Before, 0, Length, After, SubList).

sub_list(List, [], Length, Length, List, []).

sub_list([X|List], [], AccLen, Length, After, [X|SubList]) :-

    TmpLen is AccLen + 1,
    sub_list(List, [], TmpLen, Length, After, SubList).
        
    


    
