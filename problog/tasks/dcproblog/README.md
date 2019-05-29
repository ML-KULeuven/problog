# DC-ProbLog


### Installation ###
DC-ProbLog installs automatically toghether with ProbLog. However, you will need to install to further libraries that DC-ProbLog depends on additionally.

That is
1. [pytorch](https://pytorch.org/)
2. [Pyro](https://github.com/pyro-ppl/pyro)



By setting the flag `-draw_diagram` you produce a dot file that represents a compiled logic formula of the query conjoined with the evidence. If you have the 'graphviz' python package this will also immediately be rendered to a pdf.

To insall graphviz:
```
pip install graphviz
```



### How to run a program ###
You can now simply type for example:
```
problog dc example.pl
```


Use:
```
hal_problog --help
```
to find out which flags you can use to run a program



### Symbolic Probabilistic Inference Engine ###
Instead of using sampling based inference with Pyro, you can also use the exact symbolic solver that is also used in the back end of the [PSI-Solver](https://psisolver.org/). For this to work you first need to install the [psipy](https://github.com/ML-KULeuven/psipy) python package that wraps around the PSI-Solver. When running an example from the command line you then need to pass the `-psi` flag.

Note, that for large non-toy examples symbolic inference might take really long to produce any answer (depending of course how intricate your problem is).
