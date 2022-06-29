CC = g++
CFLAGS = -static -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lm -lgslcblas 
LFLAGS = -lgsl -lm -lgslcblas
INCLUDES = -I./../../env/include/
LIBS = -L./../../env/lib/

all: bin bin/line

bin:
	mkdir bin

bin/line : bin/ransampl.o src/line.cpp
	$(CC) $(CFLAGS) src/line.cpp bin/ransampl.o -o bin/line $(INCLUDES) $(LIBS) $(LFLAGS)

bin/ransampl.o : src/ransampl.c
	$(CC) $(CFLAGS) -c src/ransampl.c -o bin/ransampl.o

clean:
	rm -rf bin
