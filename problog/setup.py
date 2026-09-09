"""
problog.setup - Environment discovery
-------------------------------------

Locates the binaries and libraries ProbLog uses, and reports what is
available.  Compiling those binaries is a build-time concern and lives in
the top-level Makefile, not here.

..
    Part of the ProbLog distribution.

    Copyright 2015 KU Leuven, DTAI Research Group

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""
import shutil

import os
import sys


def get_system():
    system = sys.platform
    if system.lower().startswith("java"):
        import java.lang.System

        system = java.lang.System.getProperty("os.name").lower()
    if system.startswith("linux"):
        system = "linux"
    elif system.startswith("win"):
        system = "windows"
    elif system.startswith("mac"):
        system = "darwin"
    return system


def set_environment():
    """Updates local PATH and PYTHONPATH to include additional component directories."""
    # Set PATH environment
    path = os.environ.get("PATH", [])
    if path:
        path = [path]
    path = os.pathsep.join(path + list(get_binary_paths()))
    os.environ["PATH"] = path
    # Set PYTHONPATH environment
    sys.path += get_module_paths()


def get_binary_paths():
    """Get a list of additional binary search paths."""
    binary_root = os.path.join(os.path.dirname(__file__), "bin")
    system = get_system()  # Darwin, Linux, ?
    return list(map(os.path.abspath, [os.path.join(binary_root, system), binary_root]))


def get_module_paths():
    """Get a list of additional module search paths."""
    binary_root = os.path.join(os.path.dirname(__file__), "lib")
    system = get_system()  # Darwin, Linux, ?
    python = "python%s" % sys.version_info[0]
    return list(
        map(
            os.path.abspath,
            [
                os.path.join(binary_root, python, system),
                os.path.join(binary_root, python),
                binary_root,
            ],
        )
    )


def gather_info():
    """Collect info about the system and its installed software."""

    system_info = {}

    system_info["root_path"] = os.path.join(os.path.dirname(__file__), "..")

    system_info["os"] = get_system()
    # system_info['arch'] = os.uname()[-1]

    system_info["python_version"] = sys.version_info

    # Module pyparsing
    try:
        import pyparsing

        system_info["pyparsing"] = pyparsing.__version__
    except ImportError:
        pass

    # SDD module
    # noinspection PyBroadException
    try:
        import pysdd

        system_info["sdd_module"] = True
    except Exception:
        pass

    # DSharp
    system_info["dsharp"] = shutil.which("dsharp") is not None

    # c2d
    system_info["c2d"] = shutil.which("cnf2dDNNF") is not None
    return system_info


def system_info():
    info = gather_info()

    ok = True
    s = "System information:\n"
    s += "------------------:\n"
    s += "Operating system: %s\n" % info.get("os", "unknown")
    s += "System architecture: %s\n" % info.get("arch", "unknown")
    s += "Python version: %s.%s.%s\n" % (
        info["python_version"].major,
        info["python_version"].minor,
        info["python_version"].micro,
    )
    s += "\n"
    s += "ProbLog components:\n"
    s += "-------------------\n"

    # PrologFile, PrologString  => require pyparsing
    # SDD => requires sdd_library
    # NNF => requires dsharp or c2d

    # SemiringOther => requires NNF (or SDD with alternative evaluation)

    # pyparsing = info.get('pyparsing', 'NOT INSTALLED')
    # s += 'Module \'pyparsing\': %s\n' % pyparsing
    # if not pyparsing :
    #     s += '  ACTION: install the pyparsing module\n'
    # sdd = info.get('sdd_module', False)
    # if sdd :
    #     s += 'Module \'sdd\': INSTALLED\n'
    # else :
    #     s += 'Module \'sdd\': NOT INSTALLED\n'
    #     s += '  ACTION: run ProbLog installer\n'
    #
    return s
