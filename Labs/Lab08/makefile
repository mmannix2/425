CC=mpicc
CFLAGS=-g -Wall --std=c99

TARGET=08-deadlock
FIXED=$(TARGET)-fixed

all: $(TARGET) $(FIXED)

$(TARGET):
	$(CC) $(CFLAGS) $(TARGET).c -o $(TARGET)

$(FIXED):
	$(CC) $(CFLAGS) $(FIXED).c -o $(FIXED)

.PHONY: clean
clean:
	rm $(TARGET) $(FIXED)
