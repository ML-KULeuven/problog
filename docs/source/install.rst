Installing ProbLog
==================

Prerequisites
-------------

ProbLog is built on Python.
ProbLog is compatible with Python 3.

Python is included in most installations of Linux and Mac OSX.
Windows users can find instructions on how to install it in the
`Python documentation <https://docs.python.org/3.5/using/windows.html>`_.

Installing with pip
-------------------

ProbLog is available in the Python Package Index (PyPi) and it can be installed with


.. code-block:: bash

    pip install problog


To install as user without root permissions.

.. code-block:: bash

    pip install problog --user

After installation as user you may need to add the location of the ``problog`` script to your PATH.
Common location for this script are ``~/.local/bin`` or ``~/Library/Python/2.7/bin/``.


To update ProbLog to the latest version.

.. code-block:: bash

   pip install problog --upgrade

To install the latest ProbLog development version.

.. code-block:: bash

   pip install problog --pre --upgrade

To install ProbLog with support for Sentential Decision Diagrams (currently not supported on Windows).

.. code-block:: bash

   pip install problog[sdd]


Knowledge compilers
-------------------

ProbLog evaluates programs by compiling them to a circuit, and needs a
knowledge compiler to do it. The wheels on PyPI carry the two solvers ProbLog
builds itself, for the platform of the wheel:

* ``dsharp`` -- the default CNF to d-DNNF compiler, used by ``-k ddnnf``.
* ``maxsatz`` -- used by ``problog mpe``.

Wheels are published for Linux (glibc and musl, x86-64 and aarch64), macOS
(universal2) and Windows (x86-64). On any other platform pip falls back to the
source distribution, which is pure Python and carries no solvers. ProbLog will
then say so when a compiler is needed, and there are two ways forward: install
Sentential Decision Diagram support and use it instead,

.. code-block:: bash

    pip install problog[sdd]
    problog -k sdd model.pl

or put your own ``dsharp`` binary on your ``PATH``; ProbLog picks up whatever
it finds there in preference to the bundled one. The source is at
https://github.com/QuMuLab/dsharp.

Building the solvers from a source checkout
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The solvers are build outputs rather than committed binaries, so a checkout has
none until you build them:

.. code-block:: bash

    git submodule update --init --recursive
    make binaries

This needs a C and C++ compiler. On Windows it needs a mingw-w64 toolchain, for
example under MSYS2.

Using c2d instead of dSharp
^^^^^^^^^^^^^^^^^^^^^^^^^^^

ProbLog can use `c2d <http://reasoning.cs.ucla.edu/c2d/>`_ instead, which is
sometimes useful when dSharp struggles with a particular program. c2d is not
redistributable, so it has to be fetched separately. Download it, name the
executable ``cnf2dDNNF`` and put it on your ``PATH``; ProbLog then detects and
uses it automatically.

Only a Windows build of c2d is published. On macOS or Linux it can be run under
`Wine <https://www.winehq.org/>`_. Install Wine (``brew install wine`` on
macOS), save the Windows executable as ``cnf2dDNNF.exe``, and put this wrapper
next to it, named ``cnf2dDNNF`` and marked executable:

.. code-block:: bash

    #! /bin/bash
    wine ${BASH_SOURCE[0]}.exe $*

With both on your ``PATH``, ``problog -k ddnnf`` will use c2d.

