CC = gcc
CFLAGS = -O3 -fopenmp -fPIC -Wall -std=c99 -lm
LDFLAGS = -shared -fopenmp

TARGET = optimized_levenshtein_c.so
SRC = optimized_levenshtein_c.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
