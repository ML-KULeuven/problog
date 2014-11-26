#!/usr/bin/env python

"""
setup.py file for SWIG example
"""

from distutils.core import setup, Extension


sdd_module = Extension('_sdd',
                           sources=['sdd_wrap.c'],
                           libraries=['sdd'],
                           library_dirs=['.']
                           )

setup (name = 'sdd',
       version = '1.0',
       author      = "",
       description = """SDD Library""",
       ext_modules = [sdd_module],
       py_modules = ["sdd"],
       )
       
              #
       # PYPATH=/usr/local/Cellar/python3/3.4.2_1/Frameworks/Python.framework/Versions/3.4
       # PYTHON=python3.4
       #
       # default:
       #     swig -python sdd.i
       #     gcc -O2 -fPIC -c sdd_wrap.c -I${PYPATH}/include/${PYTHON}m
       #     gcc -shared sdd_wrap.o libsdd.a -L. -L${PYPATH}/lib/ -o _sdd.so -I${PYPATH}/include/${PYTHON}m -l${PYTHON}m
       #
       #
       # clean:
       #     rm -f _sdd.so sdd_wrap.c sdd_wrap.o sdd.py
       #