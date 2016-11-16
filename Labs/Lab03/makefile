CC=gcc
CFLAGS=-g -Wall -pthread
BROKEN=03-deadlock
FIXED=$(BROKEN)-fixed

all: $(BROKEN) $(FIXED)

$(BROKEN):
	$(CC) $(CFLAGS) -o $(BROKEN) $(BROKEN).c
$(FIXED):
	$(CC) $(CFLAGS) -o $(FIXED) $(FIXED).c
clean:
	rm $(BROKEN) $(FIXED)
