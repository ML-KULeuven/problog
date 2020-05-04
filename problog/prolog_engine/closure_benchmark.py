import sys
import os
import os.path
import random
import time
import argparse

sys.path.append("../..")

from problog.program import PrologString
from prolog_engine import PrologEngine
from problog.formula import LogicFormula
from problog.sdd_formula import SDD

def generate_program(transactions, items):
    prog = ""
    for i in range(transactions):
        for j in range(items):
            if random.choice([True, False]):
                prog += "item(" + str(i) + "," + str(j) + "). "
    prog += "\n"
    prog += "transaction(T) :- item(T,_). \n"
    prog += "item(I) :- item(_,I). \n"
    prog += "not_closed(I) :- transaction(T), item(I), \+ item(T,I). \n"
    prog += "closed(I) :- item(I), \+ not_closed(I). \n"
    return prog
    
def evaluate(prog):
    program = PrologString(prog)
    engine = PrologEngine()
    sp = engine.prepare(program)
    formula = LogicFormula(keep_all=False)
    formula = engine.ground(sp, 'closed(_)', target=formula, label=LogicFormula.LABEL_QUERY)
    sdd = SDD.create_from(formula)
    return sdd.evaluate()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("transactions", type=int)
    parser.add_argument("items", type=int)
    parser.add_argument("--out", type=str, default="closure_results.csv")
    args = parser.parse_args()
    
    prog = generate_program(args.transactions, args.items)
    start = time.time()
    evaluate(prog)
    end = time.time()
    
    new_file = not os.path.isfile(args.out)
    with open(args.out, "a") as fin:
        if new_file:
            fin.write("transactions,items,time\n")
        fin.write(str(args.transactions) + "," + str(args.items) + "," + str(end - start) + "\n")
    
if __name__ == "__main__":
    main()
