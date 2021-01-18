%Expected outcome:
% dept_salary(dept_a,1200.0) 1
% dept_salary(dept_b,950.0) 1
% dept_max_salary(dept_a,1400) 1
% dept_max_salary(dept_b,1100) 1
% dept_min_salary(dept_a,1000) 1
% dept_min_salary(dept_b,800) 1
% all_salary(1100.0) 1
% dept_all_salaries(dept_a,[1000, 1200, 1400]) 1
% dept_all_salaries(dept_b,[800, 1100]) 1


:- use_module(library(collect)).
:- use_module(library(lists)).
:- use_module(library(aggregate)).

person(a).
person(b).
person(c).
person(d).
person(e).

salary(a, 1000).
salary(b, 1200).
salary(c, 800).
salary(d, 1100).
salary(e, 1400).

dept(a, dept_a).
dept(b, dept_a).
dept(c, dept_b).
dept(d, dept_b).
dept(e, dept_a).

% Average salary per department.
collect_avg(CodeBlock, GroupBy, AggVar, AggRes) :-
    aggregate(avg, AggVar, GroupBy, CodeBlock, (GroupBy, AggRes)).

dept_salary(Dept, AvgSalary) :-
    (
        person(X),
        salary(X, Salary),
        dept(X, Dept)
    ) => Dept / avg(Salary, AvgSalary).

query(dept_salary(Dept, Salary)).

% Max salary per department.
collect_max(CodeBlock, GroupBy, AggVar, AggRes):-
    aggregate(max, AggVar, GroupBy, CodeBlock, (GroupBy, AggRes)).

dept_max_salary(Dept, MaxSalary) :-
    (
        person(X),
        salary(X, Salary),
        dept(X, Dept)
    )  => Dept/max(Salary, MaxSalary).
query(dept_max_salary(Dept, Salary)).

% Min salary per department.
collect_min(CodeBlock, GroupBy, AggVar, AggRes):-
    aggregate(min, AggVar, GroupBy, CodeBlock, (GroupBy, AggRes)).

dept_min_salary(Dept, MinSalary) :-
    (
        person(X),
        salary(X, Salary),
        dept(X, Dept)
    )  => Dept/min(Salary, MinSalary).
query(dept_min_salary(Dept, Salary)).

% Average salary company-wide.
all_salary(AvgSalary) :-
    (
        person(X),
        salary(X, Salary),
        dept(X, Dept)
    )  => avg(Salary, AvgSalary).
query(all_salary(Salary)).

%List salaries per dept
dept_all_salaries(Dept, SalariesList) :-
    (
        person(X),
        salary(X, Salary),
        dept(X, Dept)
    ) => Dept/list(Salary, SalariesList).
query(dept_all_salaries(Dept, AllSalaries)).