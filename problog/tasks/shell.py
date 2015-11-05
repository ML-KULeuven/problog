from __future__ import print_function

import sys
import readline     # provides better input

from ..program import PrologString
from ..engine import DefaultEngine
from .. import get_evaluatable
from ..util import format_dictionary
from ..core import ProbLogError
from ..logic import Term, Clause
from ..formula import LogicFormula


def show(txt):
    print (txt)


def prompt(txt='?- '):
    if sys.version_info.major == 2:
        return raw_input(txt)
    else:
        return input(txt)


def main(argv, **kwdargs):

    usage = """
    This is the interactive shell of ProbLog 2.1.

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

    """

    show('% Welcome to ProbLog 2.1')
    show('% Type \'help.\' for more information.')

    # engine = DefaultEngine()
    db = DefaultEngine().prepare([])
    knowledge = get_evaluatable()

    while True:
        try:
            cmd = prompt()
            cmd_pl = PrologString(cmd)

            for c in cmd_pl:
                if c.signature == 'listing/0':
                    print ('\n'.join(map(str, db)))
                elif c.signature == 'help/0':
                    print (usage)
                elif c.signature == 'consult/1':
                    DefaultEngine().query(db, c)
                    show('%% Consulted file %s' % c.args[0])
                elif c.signature == 'query/1':
                    gp = DefaultEngine().ground(db, c.args[0], label='query')
                    result = knowledge.create_from(gp).evaluate()
                    print (format_dictionary(result))
                else:
                    varnames = c.variables(exclude_local=True)
                    # varnames_all = c.variables(exclude_local=False)
                    #
                    # translate = {k: v for v, k in enumerate(varnames)}
                    # for v in varnames_all:
                    #     if v not in translate:
                    #         translate[v] = v
                    #
                    # c_t = c.apply(translate)

                    dbq = db.extend()
                    query_head = Term('_q', *varnames)
                    dbq += Clause(query_head, c)

                    gp = LogicFormula()
                    results = DefaultEngine().call(query_head(*range(0, len(varnames))), dbq, gp)

                    for args, node in results:
                        name = ''
                        for vn, vv in zip(varnames, args):
                            name += ('%s = %s,\n' % (vn, vv))
                        gp.add_query(Term(name), node)

                    results = knowledge.create_from(gp).evaluate()
                    for n, p in results.items():
                        print ('%sp: %s;\n---------------' % (n, p))

                    # dbq = db.extend()
                    # query_head = Term('_q', *([None] * len(varnames)))
                    # dbq += Clause(query_head, c)
                    # gp = engine.ground(dbq, query_head)


        except EOFError:
            print ('\nBye!')
            sys.exit(0)
        except KeyboardInterrupt:
            show('% CTRL-C was pressed')
        except ProbLogError as err:
            show(str(err))
        # except Exception as err:
        #     show(str(err))



def argparser():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile')
    return parser


if __name__ == '__main__':
    main(**vars(argparser().parse_args()))
