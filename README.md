ProbLog v2.1
==========

1. Prerequisites
----------------

ProbLog 2.1 requires Python 2.7+ or Python 3.2+.


2. Installation
---------------

Installation is not required.
However, if you want to use the SDD library you should run the installer.

Execute the command ``` python setup.py ```
Note that this requires the Python development files (i.e. Python.h). 


3. Usage
--------

```
usage: problog-cli.py [-h] [--knowledge {sdd,nnf}] [--symbolic]
                      [--output OUTPUT]
                      MODEL [MODEL ...]

positional arguments:
  MODEL

optional arguments:
  -h, --help            show this help message and exit
  --knowledge {sdd,nnf}, -k {sdd,nnf}
                        Knowledge compilation tool.
  --symbolic            Use symbolic evaluation.
  --output OUTPUT, -o OUTPUT
                        Output file (default stdout)
```

4. Examples
-----------

The directory 'test' contains a number of models.

* Use d-DNNF based evaluation

```
$ ./problog-cli.py test/00_trivial_and.pl -k nnf
	  heads1 : 0.5
	  heads2 : 0.6
	twoHeads : 0.3
```

* Use SDD based evaluation

```	
$ ./problog-cli.py test/00_trivial_and.pl -k sdd
	  heads1 : 0.5
	  heads2 : 0.6
	twoHeads : 0.3
```

* Use symbolic evaluation (don't compute probability) (NNF only).

```	
./problog-cli.py test/00_trivial_and.pl -k nnf --symbolic
	  heads1 : ((1-0.6)*0.5 + 0.6*0.5) / (((1-0.6)*(0.5 + (1-0.5)) + 0.6*(1-0.5)) + 0.6*0.5)
	  heads2 : (0.6*(1-0.5) + 0.6*0.5) / (((1-0.6)*(0.5 + (1-0.5)) + 0.6*(1-0.5)) + 0.6*0.5)
	twoHeads : 0.6*0.5 / (((1-0.6)*(0.5 + (1-0.5)) + 0.6*(1-0.5)) + 0.6*0.5)
```
* Evaluate all examples:

```
./problog-cli.py test/*.pl -k sdd
```
5. License
----------

The GPL GNU Public License, Version 3.0  
http://www.gnu.org/licenses/gpl-3.0.html  
Exceptions are negotiable.

Copyright (c) 2014-2015, DTAI, KU Leuven. All rights reserved.

