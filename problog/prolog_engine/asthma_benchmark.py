import sys
import os
import os.path
import random
import time
import argparse

sys.path.append("../..")

from problog.program import PrologString, SimpleProgram
from prolog_engine import PrologEngine
from problog.formula import LogicFormula
from problog.logic import Term, Constant, Clause, Var, And
from problog.sdd_formula import SDD
import subprocess
from problog.engine import DefaultEngine

def generate_program(people):
    # prog = SimpleProgram()
    prog = ''
    for i in range(people):
        prog += "person(" + str(i) + "). "
        # prog.add_fact(Term('person', Constant(i)))
        for j in range(i + 1, people):
            if random.choice([True, False]):
                prog += "friend(" + str(i) + "," + str(j) + "). "
                # prog.add_fact(Term('friend', Constant(i), Constant(j)))
    # prog.add_clause(Clause(Term('stress', Var('X')), Term('person', Var('X')), p=0.3))
    # prog.add_clause(Clause(Term('influences', Var('X'), Var('Y')), And.from_list([Term('person', Var('X')), Term('person', Var('Y')]), p=0.3)))
    prog += "0.3::stress(X) :- person(X). \n"
    prog += "0.2::influences(X,Y) :- person(X), person(Y). \n"
    prog += "smokes(X) :- stress(X). \n"
    prog += "smokes(X) :- friend(X,Y), influences(Y,X), smokes(Y). \n"
    prog += "0.4::asthma(X) :- smokes(X). \n"
    return PrologString(prog)


# def evaluate(program, engine):
#     sp = engine.prepare(program)
#     formula = engine.ground(sp, 'asthma(_)', label=LogicFormula.LABEL_QUERY)
#     # sdd = SDD.create_from(formula)
#     # return sdd.evaluate()


# def problog_prepare(prog):
#     program = prog + "query(asthma(_)). \n"
#     filename = "asthma_problog_test.pl"
#     with open(filename, "w") as fout:
#         fout.write(program)
#     return filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("people", type=int)
    parser.add_argument("--out", type=str, default="asthma_results.csv")
    args = parser.parse_args()
    default_engine = DefaultEngine()
    new_engine = PrologEngine()

    prog = generate_program(args.people)
    start = time.time()
    new_engine.ground(prog, Term('asthma', Var('X')), label=LogicFormula.LABEL_QUERY)
    stop = time.time()
    # start2 = time.time()
    # default_engine.ground(prog, Term('asthma', Var('X')), label=LogicFormula.LABEL_QUERY)
    # stop2 = time.time()
    #
    #
    #
    # new_file = not os.path.isfile(args.out)
    # with open(args.out, "a") as fin:
    #     if new_file:
    #         fin.write("method,people,time\n")
    #     fin.write("SWIProbLog," + str(args.people) + "," + str(stop - start) + "\n")
    #     fin.write("ProbLog," + str(args.people) + "," + str(stop2 - start2) + "\n")


if __name__ == "__main__":
    main()
