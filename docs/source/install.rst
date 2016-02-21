Installing ProbLog
==================

Prerequisites
-------------

ProbLog is built on Python.
ProbLog is compatible with Python 2 and 3.

Python is included in most installations of Linux and Mac OSX.
Windows users can find instructions on how to install it in the
`Python documentation <https://docs.python.org/3.5/using/windows.html>`_.

Installing Python
-----------------

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
