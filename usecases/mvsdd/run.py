"""Time categorical_sum.pl with MV-SDD and SDD for a grid of dice counts and face counts.

    python run.py [--dice 2 3 4 6 8 12 16] [--faces 2 4 8 16 32] [--timeout 30]

Every cell runs `problog categorical_sum.pl -a N -a K -k <compiler> -v` in its own process and
reports ProbLog's total time (grounding, compilation and evaluation, without starting Python).
A run that exceeds the timeout is killed; for that compiler, runs with more dice of the same
size are skipped.  Where both compilers finish, their probabilities are compared.
"""
import argparse
import os
import re
import subprocess
import sys

PROGRAM = os.path.join(os.path.dirname(os.path.abspath(__file__)), "categorical_sum.pl")
TOTAL_TIME = re.compile(r"\[INFO\] Total time: ([0-9.]+)s")
RESULT = re.compile(r"^\s*(\S+):\s+([0-9.eE+-]+)\s*$", re.MULTILINE)


def run(dice, faces, compiler, timeout):
    """Return (seconds, {query: probability}), or None when the run times out."""
    command = [sys.executable, "-m", "problog", PROGRAM, "-a", str(dice), "-a", str(faces)]
    command += ["-k", compiler, "-v"]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    seconds = TOTAL_TIME.search(done.stderr + done.stdout)
    if done.returncode != 0 or seconds is None:
        raise RuntimeError("%s failed:\n%s%s" % (" ".join(command), done.stdout, done.stderr))
    return float(seconds.group(1)), {q: float(p) for q, p in RESULT.findall(done.stdout)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dice", type=int, nargs="+", default=[2, 3, 4, 6, 8, 12, 16])
    parser.add_argument("--faces", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--timeout", type=float, default=30, help="seconds per run")
    args = parser.parse_args()

    compilers = ["mvsdd", "sdd"]
    timed_out = {c: set() for c in compilers}  # face counts that already timed out
    rows = []
    for faces in args.faces:
        for dice in args.dice:
            times = {}
            results = {}
            for compiler in compilers:
                if faces in timed_out[compiler]:
                    continue
                outcome = run(dice, faces, compiler, args.timeout)
                if outcome is None:
                    timed_out[compiler].add(faces)
                else:
                    times[compiler], results[compiler] = outcome
            if len(results) == 2:
                a, b = results["mvsdd"], results["sdd"]
                if a.keys() != b.keys() or any(abs(a[q] - b[q]) > 1e-6 for q in a):
                    raise RuntimeError("mvsdd and sdd disagree for N=%d, K=%d" % (dice, faces))
            rows.append((dice, faces, times))
            print(format_row(rows[-1], args.timeout), flush=True)


def format_row(row, timeout):
    dice, faces, times = row

    def cell(compiler):
        return "%8.3fs" % times[compiler] if compiler in times else "> %5.0fs" % timeout

    if len(times) == 2:
        speedup = "%9.1fx" % (times["sdd"] / times["mvsdd"])
    elif "mvsdd" in times:
        speedup = "> %7.0fx" % (timeout / times["mvsdd"])
    else:
        speedup = "        -"
    return "N=%-3d K=%-3d  mvsdd %s   sdd %s   speedup %s" % (
        dice, faces, cell("mvsdd"), cell("sdd"), speedup
    )


if __name__ == "__main__":
    main()
