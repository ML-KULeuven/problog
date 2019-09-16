CC = gcc
CFLAGS = -std=c99 -O2 -Wall -I../include
LIBRARY_FLAGS = -L../lib -lsdd -lm

all: test-1 test-2 test-3 test-4 test-5 \
	circuit circuit-gc

test-1: src/test-1.o
	$(CC) $(CFLAGS) $^ $(LIBRARY_FLAGS) -o $@

test-2: src/test-2.o
	$(CC) $(CFLAGS) $^ $(LIBRARY_FLAGS) -o $@

test-3: src/test-3.o
	$(CC) $(CFLAGS) $^ $(LIBRARY_FLAGS) -o $@

test-4: src/test-4.o
	$(CC) $(CFLAGS) $^ $(LIBRARY_FLAGS) -o $@

test-5: src/test-5.o
	$(CC) $(CFLAGS) $^ $(LIBRARY_FLAGS) -o $@

circuit: src/circuit.o
	$(CC) $(CFLAGS) $^ $(LIBRARY_FLAGS) -o $@

circuit-gc: src/circuit-gc.o
	$(CC) $(CFLAGS) $^ $(LIBRARY_FLAGS) -o $@

clean:
	rm -f src/*.o test-* circuit circuit-gc output/*.dot output/*.png output/*.sdd
