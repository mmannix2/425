CC = gcc
CFLAGS = -Wall -g --std=c99
CUC = nvcc
TARGET = collatz

all: $(TARGET)

$(TARGET):
	$(CUC) $(TARGET).cu -o $(TARGET)-cu
	$(CC) $(CFLAGS) $(TARGET).c -o $(TARGET)-c
clean:
	rm $(TARGET)-cu $(TARGET)-c
