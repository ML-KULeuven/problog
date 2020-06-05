from __future__ import print_function

import sys
from problog.program import PrologFile, ExtendedPrologFactory
from problog.engine import DefaultEngine
from problog.formula import LogicDAG
from problog.logic import Constant, Term
from problog.sdd_formula import SDD
from problog.cycles import break_cycles


class ConstraintFactory(ExtendedPrologFactory):
    """Extends standard ProbLog input reader to process ensure_true, ensure_false and ensure_prob."""

    def __init__(self, *args, **kwargs):
        super(ConstraintFactory, self).__init__(*args, **kwargs)
        self.constraint_count = 0

    def build_clause(self, functor, operand1, operand2, location=None, **extra):
        if len(operand1) == 1 and operand1[0].functor in (
            "ensure_true",
            "ensure_false",
            "ensure_prob",
        ):
            operand1 = [
                Term("constraint", Constant(self.constraint_count), operand1[0])
            ]
            self.constraint_count += 1
        return super(ConstraintFactory, self).build_clause(
            functor, operand1, operand2, location=location, **extra
        )


def formula_to_flatzinc_float(formula):
    """Converts a formula to a MiniZinc model using floats."""
    # print (formula)
    vardefs = []
    constraints = []
    boolvars = []

    for i, n, t in formula:
        if t == "atom":
            if str(n.probability) == "?":
                vardefs.append("var bool: B%s :: output_var;" % i)
                vardefs.append("var int: I%s;" % i)
                vardefs.append("var 0.0 .. 1.0: F%s;" % i)
                constraints.append("constraint bool2int(B%s,I%s);" % (i, i))
                constraints.append("constraint int2float(I%s,F%s);" % (i, i))
                boolvars.append("B%s" % i)
                vardefs.append("var 0.0 .. 1.0: V%s;" % i)
                vardefs.append("var 0.0 .. 1.0: Vn%s;" % i)
                constraints.append(
                    "constraint float_lin_eq([1.0, 1.0], [V%s, Vn%s], 1.0);" % (i, i)
                )
                constraints.append("constraint float_eq(F%s,V%s);" % (i, i))
            else:
                vardefs.append("float: V%s = %s;" % (i, n.probability))
                vardefs.append("float: Vn%s = %s;" % (i, 1 - float(n.probability)))
        elif t == "disj":
            vardefs.append("var 0.0 .. 1.0: V%s;" % i)
            vardefs.append("var 0.0 .. 1.0: Vn%s;" % i)
            constraints.append(
                "constraint float_lin_eq([1.0, 1.0], [V%s, Vn%s], 1.0);" % (i, i)
            )

            children = ["V%s" % str(c).replace("-", "n") for c in n.children] + [
                "V%s" % i
            ]
            coef = ["1.0"] * (len(children) - 1) + ["-1.0"]
            constraints.append(
                "constraint float_lin_eq([%s], [%s], 0.0);"
                % (", ".join(coef), ", ".join(children))
            )
        elif t == "conj":
            vardefs.append("var 0.0 .. 1.0: V%s;" % i)
            vardefs.append("var 0.0 .. 1.0: Vn%s;" % i)
            constraints.append(
                "constraint float_lin_eq([1.0, 1.0], [V%s, Vn%s], 1.0);" % (i, i)
            )

            children = [str(c).replace("-", "n") for c in n.children]
            constraints.append(
                "constraint float_times(V%s, V%s, V%s);" % (children[0], children[1], i)
            )

    for name, index in formula.get_names(label="constraint"):
        index = str(index).replace("-", "n")
        con = name.args[1]
        if con.functor == "ensure_prob":
            constraints.append(
                "constraint float_lin_le([1.0],[V%s], %s);" % (index, con.args[1])
            )
            constraints.append(
                "constraint float_lin_le([-1.0], [V%s], -%s);" % (index, con.args[0])
            )
        elif con.functor == "ensure_true":
            constraints.append("constraint float_eq(V%s, 1.0);" % (index,))
        elif con.functor == "ensure_false":
            constraints.append("constraint float_eq(V%s, 0.0);" % (index,))

    solve = [
        "solve :: bool_search([%s], input_order, indomain_min, complete) satisfy;"
        % ", ".join(boolvars)
    ]

    fzn = "\n".join(vardefs + constraints + solve)

    return fzn


def formula_to_flatzinc_bool(formula):
    """Converts a formula to a MiniZinc model using booleans."""
    vardefs = []
    constraints = []
    boolvars = []

    for i, n, t in formula:
        if t == "atom":
            if str(n.probability) == "?":
                vardefs.append("var bool: B%s :: output_var;" % i)
            else:
                vardefs.append("bool: B%s = true;" % (i,))
        elif t == "disj":
            vardefs.append("var bool: B%s;" % i)
            children = ["B%s" % str(c).replace("-", "n") for c in n.children]
            constraints.append(
                "constraint array_bool_or([%s], %s);" % (", ".join(children), "B%s" % i)
            )
        elif t == "conj":
            vardefs.append("var bool: B%s;" % i)
            children = ["B%s" % str(c).replace("-", "n") for c in n.children]
            constraints.append(
                "constraint array_bool_and([%s], %s);"
                % (", ".join(children), "B%s" % i)
            )
        vardefs.append("var bool: Bn%s;" % i)
        constraints.append("constraint bool_not(B%s, Bn%s);" % (i, i))

    for name, index in formula.get_names(label="constraint"):
        if index is not None:
            index = str(index).replace("-", "n")
            con = name.args[1]
            assert con.functor != "ensure_prob"
            if con.functor == "ensure_true":
                constraints.append("constraint bool_eq(B%s, true);" % (index,))
            elif con.functor == "ensure_false":
                constraints.append("constraint bool_eq(B%s, false);" % (index,))

    solve = [
        "solve :: bool_search([%s], input_order, indomain_min, complete) satisfy;"
        % ", ".join(boolvars)
    ]

    fzn = "\n".join(vardefs + constraints + solve)

    return fzn


def debug(*args):
    """Helper function for printing debugging output."""
    print(*args, file=sys.stderr)


def solve(fzn):
    """Call FlatZinc solver (Gecode) and process output."""
    import subprocess

    from problog.util import mktempfile

    fznfile = mktempfile(".fzn")
    with open(fznfile, "w") as f:
        f.write(fzn)
    cmd = ["fzn-gecode", "-a", fznfile]
    result = subprocess.check_output(cmd).decode()

    current = {}
    for line in result.split("\n"):
        if line.startswith("----"):
            yield current
            current = {}
        elif line.startswith("===="):
            return
        else:
            vr, vl = line.strip(";").split(" = ")
            vr = int(vr[1:])
            vl = 1 if vl == "true" else 0
            current[vr] = vl


def compress(formula, atoms):
    """Compress the formula by setting the values of the given atoms."""

    # TODO support cyclic programs

    formula.clear_labeled("constraint")

    out = LogicDAG()

    relevant = formula.extract_relevant()

    cyclic = set()

    translate = {}
    for i, n, t in formula:

        if not relevant[i]:
            continue
        elif t == "atom":
            if i in atoms:
                if atoms[i] == 1:
                    translate[i] = 0
                else:
                    translate[i] = None
            else:
                translate[i] = out.add_atom(*n)
        else:
            children = []
            for c in n.children:
                if abs(c) in translate:
                    tr = translate[abs(c)]
                    if c > 0:
                        children.append(tr)
                    else:
                        children.append(out.negate(tr))
                else:
                    cyclic.add(c)
            if t == "conj":
                translate[i] = out.add_and(children)
            else:
                translate[i] = out.add_or(children)

    if cyclic:
        debug("Original formula:\n", formula)
        debug("Cyclic nodes:", cyclic)
        debug("New formula so far:\n", out)
        raise RuntimeError(
            "Cycle detected in program. Cycles are currently not supported."
        )

    for q, n, l in formula.labeled():
        if l != "constraint":
            if n < 0:
                out.add_name(q, out.negate(translate[-n]), l)
            else:
                out.add_name(q, translate[n], l)
    return out


def main(argv, handle_output=None):
    args = argparser().parse_args(argv)

    verbose = args.verbose
    filename = args.filename

    if verbose:
        debug("Loading...")
    problog_model = PrologFile(filename, factory=ConstraintFactory())

    engine = DefaultEngine()
    database = engine.prepare(problog_model)

    # Ground the constraints
    if verbose:
        debug("Grounding...")
    target = engine.ground(database, Term("constraint", None, None), label="constraint")

    queries = [q[0] for q in engine.query(database, Term("query", None))]
    for query in queries:
        target = engine.ground(database, query, label="query", target=target)

    if verbose > 1:
        print(target, file=sys.stderr)

    has_prob_constraint = False
    for name, index in target.get_names(label="constraint"):
        if index is not None:
            if name.args[1].functor == "ensure_prob":
                has_prob_constraint = True

    # Compile and turn into CP-problem
    if has_prob_constraint:
        if verbose:
            debug("Probabilistic constraints detected.")

        if verbose:
            debug("Compiling...")
        sdd = SDD.create_from(target)
        formula = sdd.to_formula()
        # Convert to flatzinc
        if verbose:
            debug("Converting...")
        fzn = formula_to_flatzinc_float(formula)
        sdd.clear_labeled("constraint")

        if verbose > 1:
            print(fzn, file=sys.stderr)

        # Solve the flatzinc model
        if verbose:
            debug("Solving...")
        sols = list(solve(fzn))

        has_solution = False
        for i, res in enumerate(sols):
            if verbose:
                debug("Evaluating solution %s/%s..." % (i + 1, len(sols)))
            # target.lookup_evidence = {}
            weights = sdd.get_weights()
            # ev_nodes = []
            for k, v in res.items():
                sddvar = formula.get_node(k).identifier
                sddatom = sdd.var2atom[sddvar]
                sddname = sdd.get_name(sddatom)
                weights[sddatom] = v
            for k, v in sdd.evaluate().items():
                if v > 0.0:
                    print(k, v)
            print("----------")
            has_solution = True
        if has_solution:
            print("==========")
        else:
            print("=== UNSATISFIABLE ===")

    else:
        formula = LogicDAG()
        break_cycles(target, formula)

        if verbose:
            debug("Converting...")
        fzn = formula_to_flatzinc_bool(formula)

        if verbose > 1:
            print(fzn, file=sys.stderr)

        if verbose:
            debug("Solving...")
        sols = list(solve(fzn))

        has_solution = False
        for i, res in enumerate(sols):
            if verbose:
                debug("Evaluating solution %s/%s..." % (i + 1, len(sols)))

            if verbose:
                debug("Compressing...")
            new_formula = compress(formula, res)

            if verbose:
                debug("Compiling...")
            sdd = SDD.create_from(new_formula)

            if verbose:
                debug("Evaluating...")
            for k, v in sdd.evaluate().items():
                if v > 0.0:
                    print(k, v)
            print("----------")
            has_solution = True
        if has_solution:
            print("==========")
        else:
            print("=== UNSATISFIABLE ===")


def argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", metavar="MODEL")
    parser.add_argument("-v", "--verbose", action="count")
    return parser


if __name__ == "__main__":
    main(sys.argv[1:])
