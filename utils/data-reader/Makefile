# The compilers to use.
CC = g++

# Directories for Includes and Common clases
IDIR =-I../include/   -I/usr/include/boost/
CDIR =../plib/
LDIR =-L/usr/lib64/
LIBS = -lboost_thread

# Compiler options
CFLAGS= -c -m64 -O3 -Wall $(IDIR)
LDFLAGS= $(LIBS) $(LDIR) $(LIBS)

# Sources and Common clases sources
SOURCES= main.cpp StringTokenizer.cpp

# Objects
OBJECTS=$(SOURCES:.cpp=.o)

# Name of the executable
EXECUTABLE=mrmr-reader

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o: $(SOURCES) $(CSOURCES)
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm *.o
	rm mrmr-reader
