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