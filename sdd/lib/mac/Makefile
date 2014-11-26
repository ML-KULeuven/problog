

# PYPATH=/usr/local/Cellar/python3/3.4.2_1/Frameworks/Python.framework/Versions/3.4
PYTHON=python

default:
	swig -python sdd.i
	${PYTHON} setup.py build_ext --inplace
	
# gcc -O2 -fPIC -c sdd_wrap.c -I${PYPATH}/include/${PYTHON}m
# gcc -shared sdd_wrap.o libsdd.a -L. -L${PYPATH}/lib/ -o _sdd.so -I${PYPATH}/include/${PYTHON}m -l${PYTHON}m


clean:
	rm -f _sdd.so sdd_wrap.c sdd_wrap.o sdd.py
