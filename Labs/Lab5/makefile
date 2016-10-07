CC=gcc
CFLAGS=-g -Wall -fopenmp -lm
SERIAL=05-factor
PARALLEL=$(SERIAL)-omp

all: $(SERIAL) $(PARALLEL)

$(SERIAL):
	$(CC) -o $(SERIAL) $(SERIAL).c $(CFLAGS)

$(PARALLEL):
	$(CC) -o $(PARALLEL) $(PARALLEL).c $(CFLAGS)

clean:
	rm $(SERIAL) $(PARALLEL)
