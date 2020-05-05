import sys
import os
import os.path
import random
import time
import argparse

sys.path.append("../..")
from problog.engine import DefaultEngine
from problog.program import PrologString
from prolog_engine import PrologEngine
from problog.formula import LogicFormula
from problog.logic import Term, Var
from problog.sdd_formula import SDD
import subprocess

def generate_program(people):
    prog = ""
    for i in range(people):
        prog += "person(" + str(i) + "). "
        for j in range(i+1, people):
            if random.choice([True, False]):
                prog += "friend(" + str(i) + "," + str(j) + "). "
    prog += "\n"
    prog += "0.3::stress(X) :- person(X). \n"
    prog += "0.2::influences(X,Y) :- person(X), person(Y). \n"
    prog += "smokes(X) :- stress(X). \n"
    prog += "smokes(X) :- friend(X,Y), influences(Y,X), smokes(Y). \n"
    prog += "0.4::asthma(X) :- smokes(X). \n"
    return PrologString(prog)
    
def evaluate(prog, engine_Class):
    # program = PrologString(prog)
    engine = engine_Class()
    sp = engine.prepare(prog)
    formula = LogicFormula(keep_all=False)
    start = time.time()
    formula = engine.ground(sp, Term('asthma',Var('X')), target=formula, label=LogicFormula.LABEL_QUERY, profile=1)
    ground_time = time.time()-start
    print(engine)
    print(ground_time)
    # sdd = SDD.create_from(formula)
    # return sdd.evaluate()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("people", type=int)
    parser.add_argument("--out", type=str, default="asthma_results.csv")
    args = parser.parse_args()
    
    prog = generate_program(args.people)
    # for engine in [DefaultEngine, PrologEngine]:
    for engine in [PrologEngine]:
        evaluate(prog, engine)
    # start = time.time()
    # evaluate(prog)
    # stop = time.time()
#
#     filename = problog_prepare(prog)
#     start2 = time.time()
#     subprocess.call(["python3", "/home/yann/master_problog/problog-cli.py", "ground", "--break-cycles", filename, "--output", "/dev/null"])
#     stop2 = time.time()
#
#     new_file = not os.path.isfile(args.out)
#     with open(args.out, "a") as fin:
#         if new_file:
#             fin.write("method,people,time\n")
#         fin.write("SWIProbLog," + str(args.people) + "," + str(stop - start) + "\n")
#         fin.write("ProbLog," + str(args.people) + "," + str(stop2 - start2) + "\n")

if __name__ == "__main__":
    main()
