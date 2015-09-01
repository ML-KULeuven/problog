ProbLog v2.1
==========

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
$ python problog-cli.py test/00_trivial_and.pl -k nnf
	  heads1 : 0.5
	  heads2 : 0.6
	twoHeads : 0.3
```

* Use SDD based evaluation (not available on Windows)

```
$ python problog-cli.py test/00_trivial_and.pl -k sdd
	  heads1 : 0.5
	  heads2 : 0.6
	twoHeads : 0.3
```

* Use symbolic evaluation (don't compute probability) (NNF only).

```
python problog-cli.py test/00_trivial_and.pl -k nnf --symbolic
	  heads1 : ((1-0.6)*0.5 + 0.6*0.5) / (((1-0.6)*(0.5 + (1-0.5)) + 0.6*(1-0.5)) + 0.6*0.5)
	  heads2 : (0.6*(1-0.5) + 0.6*0.5) / (((1-0.6)*(0.5 + (1-0.5)) + 0.6*(1-0.5)) + 0.6*0.5)
	twoHeads : 0.6*0.5 / (((1-0.6)*(0.5 + (1-0.5)) + 0.6*(1-0.5)) + 0.6*0.5)
```
* Evaluate all examples:

```
python problog-cli.py test/*.pl -k sdd
```
5. License
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
