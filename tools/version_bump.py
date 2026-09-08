#!/usr/bin/env python3
"""Bump problog/version.py.  Previously lived in setup.py."""

import os
import sys

VERSION_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "problog", "version.py"
)


def read_version():
    ns = {}
    with open(VERSION_FILE) as fp:
        exec(fp.read(), ns)
    return ns["version"]


def increment_release(v):
    v = v.split(".")
    if len(v) == 4:
        v = v[:3] + [str(int(v[3]) + 1)]
    else:
        v = v[:4]
    return ".".join(v)


def increment_dev(v):
    v = v.split(".")
    if len(v) == 4:
        v = v[:3] + [str(int(v[3]) + 1), "dev1"]
    else:
        v = v[:4] + ["dev" + str(int(v[4][3:]) + 1)]
    return ".".join(v)


def write_version(v):
    with open(VERSION_FILE, "w") as f:
        f.write("version = '%s'\n" % v)
    print(v)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "show"
    current = read_version()
    if mode == "show":
        print(current)
    elif mode == "release":
        write_version(increment_release(current))
    elif mode == "dev":
        write_version(increment_dev(current))
    else:
        sys.exit("usage: version_bump.py [show|release|dev]")
