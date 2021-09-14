import atexit
import os
import sys

try:
    import readline  # provides better input
except:
    readline = None

from ..program import PrologString
from ..engine import DefaultEngine
from .. import get_evaluatable
from ..util import format_dictionary
from ..core import ProbLogError
from ..logic import Term, Clause, term2str
from ..formula import LogicFormula
from ..version import version


def show(txt):
    print(txt)


def prompt(txt="?- "):
    if sys.version_info.major == 2:
        return raw_input(txt)
    else:
        return input(txt)


usage = f"""
    This is the interactive shell of ProbLog {version}.

    You probably want to load a program first:

        ?- consult('test/3_tossing_coin.pl').

    You can show the currently loaded code using 'listing'.

        ?- listing.

    You can execute ProbLog queries by simpling typing them as you would in Prolog:

        ?- heads(X).
        X = c4,
        p: 0.6;
        ---------------
        X = c3,
        p: 0.6;
        ---------------
        X = c2,
        p: 0.6;
        ---------------
        X = c1,
        p: 0.6;
        ---------------

    You can also type more complex queries:

        ?- heads(X), heads(Y), X \= Y.
        X = c4,
        Y = c1,
        p: 0.36;
        ---------------
        X = c4,
        Y = c2,
        p: 0.36;
        ---------------
        ...

    Evidence can be provided using a pipe (|):

        ?- someHeads | not heads(c1).
        p: 0.936;
        ---------------

    Note that variables are not shared between query and evidence.
    
    You can set shell options with the predicate 
    
        option(Option, Value)
        
    The following options are supported:
    
        show_zero  no/yes   show query results with probability 0.0
        
    You can check the currently set value by passing a variable as the second argument.

    """


class Option(object):
    pass


class BooleanOption(Option):
    def __init__(self, default=True):
        self.value = default

    def set(self, value):
        if value.is_var():
            if value.name[0] != "_":
                return "%s = %s" % (value, self.get())
            else:
                return None
        elif value.signature == "yes/0":
            self.value = True
        elif value.signature == "no/0":
            self.value = False

    def get(self):
        return "yes" if self.value else "no"

    def __nonzero__(self):
        return self.value


def main(argv, **kwdargs):
    if readline:
        histfile = os.path.join(os.path.expanduser("~"), ".probloghistory")
        try:
            readline.read_history_file(histfile)
        except IOError:
            pass
        atexit.register(readline.write_history_file, histfile)

    show("%% Welcome to ProbLog 2.2 (version %s)" % version)
    show("% Type 'help.' for more information.")

    # engine = DefaultEngine()
    db = DefaultEngine().prepare([])
    knowledge = get_evaluatable()

    nonprob = ["consult/1", "use_module/1"]

    options = {Term("show_zero"): BooleanOption(False)}

    while True:
        try:
            cmd = prompt()

            cmd_pl = PrologString(cmd)

            for c in cmd_pl:
                if c.signature == "listing/0":
                    print("\n".join(map(str, db)))
                elif c.signature == "help/0":
                    print(usage)
                elif c.signature == "option/2":
                    try:
                        result = options[c.args[0]].set(c.args[1])
                        if result is not None:
                            print(result)
                    except KeyError:
                        raise ProbLogError("Unknown option '%s'" % c.args[0])
                elif c.signature in nonprob:
                    DefaultEngine().query(db, c)
                    db = DefaultEngine().prepare(db)
                    # show('%% Consulted file %s' % c.args[0])
                elif c.signature == "query/1":
                    gp = DefaultEngine().ground(db, c.args[0], label="query")
                    result = knowledge.create_from(gp).evaluate()
                    print(format_dictionary(result))
                else:
                    gp = LogicFormula()
                    dbq = db.extend()
                    if c.functor == "'|'":
                        ev_c = c.args[1]
                        c = c.args[0]
                        varnames = ev_c.variables(exclude_local=True)
                        query_head = Term("_e", *varnames)
                        dbq += Clause(query_head, ev_c)
                        results = DefaultEngine().call(
                            query_head(*range(0, len(varnames))), dbq, gp
                        )

                        for args, node in results:
                            name = ""
                            for vn, vv in zip(varnames, args):
                                name += "%s = %s,\n" % (vn, vv)
                            gp.add_evidence(Term(name), node, True)

                    varnames = c.variables(exclude_local=True)
                    query_head = Term("_q", *varnames)
                    dbq += Clause(query_head, c)

                    results = DefaultEngine().call(
                        query_head(*range(0, len(varnames))), dbq, gp
                    )

                    for args, node in results:
                        name = ""
                        for vn, vv in zip(varnames, args):
                            name += "%s = %s,\n" % (vn, term2str(vv))
                        gp.add_query(Term(name), node)

                    results = knowledge.create_from(gp).evaluate()
                    for n, p in results.items():
                        if p > 0 or options[Term("show_zero")]:
                            print("%sp: %s;\n---------------" % (n, p))

                    # dbq = db.extend()
                    # query_head = Term('_q', *([None] * len(varnames)))
                    # dbq += Clause(query_head, c)
                    # gp = engine.ground(dbq, query_head)

        except EOFError:
            print("\nBye!")
            sys.exit(0)
        except KeyboardInterrupt:
            show("% CTRL-C was pressed")
        except ProbLogError as err:
            show(str(err))
        # except Exception as err:
        #     show(str(err))


def argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile")
    return parser


if __name__ == "__main__":
    main(**vars(argparser().parse_args()))
