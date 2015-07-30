"""
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
from __future__ import print_function

import os

def build_sdd():

    build_lib = '.'  # where to put final library?
    build_dir = '.'  # where to put temporary build files?

    lib_dir = '.'   # where to find SDD library

    curr = os.curdir
    os.chdir(build_dir)

    from distutils.core import setup, Extension
    sdd_module = Extension('_sdd', sources=['sdd_wrap.c', 'except.c'], libraries=['sdd'], library_dirs=[lib_dir])

    setup (name='sdd',
           version='1.0',
           author="",
           description="""SDD Library""",
           ext_modules=[sdd_module],
           py_modules=["sdd"],
           script_name='',
           script_args=['build_ext', '--build-lib', build_lib, '--rpath', lib_dir]
    )

    os.chdir(curr)

if __name__ == '__main__':
    build_sdd()
