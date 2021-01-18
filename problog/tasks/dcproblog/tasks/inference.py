import argparse

from problog import library_paths, root_path
from problog.errors import InstallError
from problog.program import PrologFile
from ..solver import InferenceSolver


def argparser(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("file_name", type=str, help="HAL-ProbLog file")
    parser.add_argument(
        "-draw_diagram", help="draw and save SDD diagram .gv/.pdf", action="store_true"
    )
    parser.add_argument("-dpath", type=str, help="where to save diagram")

    if "-psi" in args:
        parser.add_argument(
            "-psi",
            help="use the PSI-Solver",
            dest="abe_name",
            const="psi",
            action="store_const",
        )
    elif "-pyro" in args:
        parser.add_argument(
            "-pyro",
            help="use the pyro library (default)",
            dest="abe_name",
            const="pyro",
            action="store_const",
        )
        parser.add_argument("-n_samples", type=int, help="number of samples")
        parser.add_argument(
            "-ttype", type=str, help="type of tensor (float32 or float64)"
        )
        parser.add_argument(
            "-device",
            type=str,
            help="on which device you run, e.g. cpu (default), cuda, cuda:0",
        )

        parser.set_defaults(device="cpu")
        parser.set_defaults(ttype="float32")
        parser.set_defaults(n_samples=2000)
    else:
        parser.set_defaults(abe_name="pyro")
        parser.add_argument("-n_samples", type=int, help="number of samples")
        parser.add_argument(
            "-ttype", type=str, help="type of tensor: float32 (default) or float64"
        )
        parser.add_argument(
            "-device",
            type=str,
            help="on which device you run, e.g. cpu (default), cuda, cuda:0",
        )

        parser.set_defaults(device="cpu")
        parser.set_defaults(ttype="float32")
        parser.set_defaults(n_samples=2000)

    return parser


def main(args):
    library_paths.append(root_path("problog", "tasks", "dcproblog", "library"))

    import time

    start = time.time()
    parser = argparser(args)
    args = parser.parse_args(args)
    args = vars(args)

    if args["abe_name"] == "pyro":
        try:
            import pyro
        except:
            raise InstallError("Pyro is not available.")
    elif args["abe_name"] == "psi":
        try:
            import psi
        except:
            raise InstallError("pypsi is not available.")

    program = PrologFile(args["file_name"])

    solver = InferenceSolver(**args)
    probabilities = solver.probability(program, **args)
    # print("time: {final}".format(final=time.time()-start))
    solver.print_result(probabilities)
