ProbLog v2.1
============

1. Prerequisites
----------------

ProbLog 2.1 requires Python 2.7+ or Python 3.2+.


2. Installation
---------------

ProbLog 2.1 works out of the box on systems with Python.
It has been tested on Mac OSX, Linux and Windows.

ProbLog supports optional components which can be installed separately.
See the file INSTALL for detailed installation instructions.

3. Usage
--------

See documentation_.

.. _documentation: http://problog.readthedocs.org/en/latest/cli.html

4. License
----------

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

SC-ProbLog
==========

1. Prerequisites
----------------
SC-ProbLog requires ProbLog v2.1, Python 3.4 and Gurobi 6.5.2. 

2. Installation
---------------
Clone repository and checkout sc-problog branch.
Install SC-ProbLog by running
$ python3.4 setup.py
in the problog subdirectory.

3. Usage
--------
The subdirectory 'examples/sc-problog examples' contains a minimal
working example with a small experiment. 

Different formulations of a Viral Marketing problem are evaluated on 
a small social network.
The MIP solver used in this example is Gurobi, and its performance
is evaluated on SDDs that are not minimized, and those that are 
minimized by the custom SMP minimization algorithm.

More details on experimental settings etc can be found in the comments
in the example file.

Note that the minimization type (default or SMP) now still has to be
set manually in sdd_formula.py (line 59) in the problog subdirectory,
after which the setup.py file has to be run again to reflect the changes.

4. License
----------
Same as for ProbLog 2.1.

5. Future
---------
More examples and more flexible support for solving SCOPs with Gecode
as well as Gurobi will be added in the future.