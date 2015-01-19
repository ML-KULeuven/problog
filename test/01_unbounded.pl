% %Expected outcome:
% % ERROR CallStackError
%
% 0.3::p(1).
% 0.2::p(2).
% 0.1::p(3).
% 0.5::p(X) :- X2 is X-1, p(X2).
%
% a(1).
% a(2).
% a(3).
% a(4).
% a(5).
%
% query(p(X)) :- a(X).
%
