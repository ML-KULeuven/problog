
# ProbLog

ProbLog 2 is a **Probabilistic Logic Programming toolbox**.
It allows to intuitively build programs that do **not only encode complex interactions** between a large sets of heterogenous components,
but also the **inherent uncertainties** that are present in real-life situations.

Probabilistic logic programs are **logic programs** in which some of the facts are annotated with **probabilities**.

The engine tackles several tasks such as **computing the marginals given evidence** and **learning from (partial) interpretations**.
ProbLog is a suite of efficient algorithms for various **inference tasks**.
It is based on a conversion of the program and the queries and evidence to a **weighted Boolean formula**.
This allows us to reduce the inference tasks to well-studied tasks such as **weighted model counting**,
which can be solved using state-of-the-art methods known from the **graphical model** and **knowledge compilation literature**.

ProbLog is a **Python** package and can be embedded in Python or Java.
Its knowledge base can be represented as **Prolog/Datalog** facts, **CSV-files**, **SQLite** database tables,
through functions implemented in the host environment or combinations hereof.

ProbLog 2.1 works out of the box on systems with Python. It has been
tested on *Mac OSX*, *Linux* and *Windows*.
## Installation

[![CircleCI](https://circleci.com/gh/ML-KULeuven/problog/tree/master.svg?style=svg)](https://circleci.com/gh/ML-KULeuven/problog/tree/master)
[![codecov](https://codecov.io/gh/ML-KULeuven/problog/branch/master/graph/badge.svg)](https://codecov.io/gh/ML-KULeuven/problog)
dev:
[![CircleCI](https://circleci.com/gh/ML-KULeuven/problog/tree/develop.svg?style=svg)](https://circleci.com/gh/ML-KULeuven/problog/tree/develop)
[![codecov](https://codecov.io/gh/ML-KULeuven/problog/branch/develop/graph/badge.svg)](https://codecov.io/gh/ML-KULeuven/problog)


ProbLog supports optional components which can be installed separately.
See the file [INSTALL](https://github.com/ML-KULeuven/problog/blob/develop/INSTALL) for detailed installation instructions.

### Python

To install ProbLog, you can use the [pip](https://pypi.org/project/pip/) with the following command:

```pip install problog```

#### Prerequisites

ProbLog 2.1 requires Python 3.6+.
*(Python 2.7+ support has been dropped since ProbLog 2.1.0.36.)*


### Online Editor

You can try out ProbLog without installing it with our [online editor](https://dtai.cs.kuleuven.be/problog/editor.html).

For example, enter the following [ProbLog program](https://dtai.cs.kuleuven.be/problog/editor.html#task=prob&hash=f8cdb15e6accf62ecaf706c230197ce1) for calculating the probability that at least one of two coins
(one of which is a bend/biased coin) is head.

```
% Probabilistic facts:
0.5::heads1.
0.6::heads2.

% Rules:
someHeads :- heads1.
someHeads :- heads2.

% Queries:
query(someHeads).
```

When you press evaluate, this will result in **0.8**,
because *P(someHeads) = 1 - (1-P(heads1)) (1-P(heads2)) = 1 - (1-0.5) (1-0.6) = 0.8*.
 


## Get Started with ProbLog

### Tutorial
To get started with ProbLog, follow the [ProbLog Tutorial](https://dtai.cs.kuleuven.be/problog/tutorial.html).

### Homepage

Visit the [ProbLog Homepage](https://dtai.cs.kuleuven.be/problog/).

### Documentation
Extensive documentation about ProbLog can be found on our 
[ProbLog documentation](http://problog.readthedocs.org/en/latest/cli.html) on ReadTheDocs.

### Papers

You can consult the following paper to get an introduction to ProbLog:

[Inference and learning in probabilistic logic programs using weighted Boolean formulas](https://lirias.kuleuven.be/bitstream/123456789/392821/3/plp2cnf.pdf), Daan Fierens, Guy Van den Broeck, Joris Renkens, Dimitar Shterionov, Bernd Gutmann, Ingo Thon, Gerda Janssens, and Luc De Raedt. Theory and Practice of Logic Programming, 2015. 

[ProbLog: A probabilistic Prolog and its application in link discovery](https://lirias.kuleuven.be/bitstream/123456789/146072/1/ijca), L. De Raedt, A. Kimmig, and H. Toivonen, Proceedings of the 20th International Joint Conference on Artificial Intelligence (IJCAI-07), Hyderabad, India, pages 2462-2467, 2007. 

Many other papers and information about ProbLog can be found in [our ProbLog publication list](https://dtai.cs.kuleuven.be/problog/publications.html).


## License

Copyright 2015 KU Leuven, DTAI Research Group

Licensed under the Apache License, Version 2.0 (the "License"); you may
not use this file except in compliance with the License. You may obtain
a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.