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
SOURCES= main.cpp
CSOURCES= $(CDIR)Histogram.cpp $(CDIR)JointProb.cpp $(CDIR)MutualInfo.cpp $(CDIR)ProbTable.cpp $(CDIR)RawData.cpp

# Objects
OBJECTS=$(SOURCES:.cpp=.o)
COBJECTS=$(CSOURCES:.cpp=.o)

# Name of the executable
EXECUTABLE=main

all: $(SOURCES) $(CSOURCES) $(CUSOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) $(COBJECTS) $(CUOBJECTS)
	$(CC) $(OBJECTS) $(COBJECTS) $(CUOBJECTS) $(LDFLAGS) -o $@

.cpp.o: $(SOURCES) $(CSOURCES)
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm *.o
	rm $(CDIR)*.o
	rm main
