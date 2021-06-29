"""

..
    Part of the ProbLog distribution.

    Copyright 2015 KU Leuven, DTAI Research Group

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        https://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import argparse
import json
import sys
import traceback

from ..engine import DefaultEngine
from ..errors import process_error
from ..kbest import KBestFormula
from ..program import PrologFile
from ..util import init_logger, format_dictionary


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("-v", "--verbose", action="count")
    parser.add_argument("--web", action="store_true")
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args(argv)

    init_logger(args.verbose)
    if args.output:
        out = open(args.output, "w")
    else:
        out = sys.stdout

    result = {}
    try:
        pl = PrologFile(args.filename)
        db = DefaultEngine().prepare(pl)

        program = list(map(lambda s: "%s." % s, db.iter_raw()))
        cnf = KBestFormula.create_from(db, label_all=True)

        explanation = []
        results = cnf.evaluate(explain=explanation)

        result["SUCCESS"] = True
        result["program"] = program
        result["proofs"] = explanation
        result["results"] = results
        if args.web:
            result["probabilities"] = [
                (str(k), round(v, 8)) for k, v in results.items()
            ]
            print(json.dumps(result), file=out)
        else:
            print("Transformed program", file=out)
            print("-------------------", file=out)
            print("\n".join(program), file=out)
            print(file=out)

            print("Proofs", file=out)
            print("------", file=out)
            print("\n".join(explanation), file=out)

            print(file=out)
            print("Probabilities", file=out)
            print("-------------", file=out)
            print(format_dictionary(results), file=out)

    except Exception as err:
        trace = traceback.format_exc()
        err.trace = trace
        if args.web:
            result["SUCCESS"] = False
            result["err"] = vars(err)
            result["err"]["message"] = process_error(err)
            print(json.dumps(result), file=out)
        else:
            print(process_error(err), file=out)

    if args.output:
        out.close()

    return result


if __name__ == "__main__":
    main(sys.argv[1:])


def argparser():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile")
    return parser


if __name__ == "__main__":
    main(**vars(argparser().parse_args()))
