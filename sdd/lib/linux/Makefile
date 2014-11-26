
default:
	swig -python sdd.i
	gcc -fpic -c sdd_wrap.c -I/usr/include/python3.2mu/
	ld -shared sdd_wrap.o -Llib -lsdd -lm -lpython3.2mu -XLinker -rpath lib -o _sdd.so

clean:
	rm -f _sdd.so sdd_wrap.c sdd_wrap.o sdd.py

