%Expected outcome:
% dept_salary(dept_a,1200.0) 1
% dept_salary(dept_b,950.0) 1
% dept_max_salary(dept_a,1400) 1
% dept_max_salary(dept_b,1100) 1
% dept_min_salary(dept_a,1000) 1
% dept_min_salary(dept_b,800) 1
% all_salary(1100.0) 1


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
dept_salary(Dept, avg<Salary>) :- person(X), salary(X, Salary), dept(X, Dept).
query(dept_salary(Dept, Salary)).

% Max salary per department.
dept_max_salary(Dept, max<Salary>) :- person(X), salary(X, Salary), dept(X, Dept).
query(dept_max_salary(Dept, Salary)).

% Min salary per department.
dept_min_salary(Dept, min<Salary>) :- person(X), salary(X, Salary), dept(X, Dept).
query(dept_min_salary(Dept, Salary)).

% Average salary company-wide.
all_salary(avg<Salary>) :- person(X), salary(X, Salary), dept(X, Dept).
query(all_salary(Salary)).