CC=mpicc
CFLAGS=-g -Wall --std=c99
TARGET=test

P2P=$(TARGET)-p2p
BCAST=$(TARGET)-bcast

all: $(P2P) $(BCAST)

$(P2P):
	$(CC) $(CFLAGS) $(P2P).c -o $(P2P)

$(BCAST):
	$(CC) $(CFLAGS) $(BCAST).c -o $(BCAST)

.PHONY: clean
clean:
	rm $(P2P) $(BCAST)
