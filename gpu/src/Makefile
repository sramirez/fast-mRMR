#Set the cuda variables and boost directories properly before using
# Cuda variables
CUDAPATH = /usr/local/cuda-5.5
GPUARCH = sm_20

# Boost directories
BOOST_INCLUDE = /usr/include/boost/
BOOST_LIB = /usr/lib64/

# The compilers to use.
CC = g++
NVCC = nvcc

# Directories for Includes and Common clases
IDIR =-I../include/ -I$(BOOST_INCLUDE) -I$(CUDAPATH)/include/
CDIR =../plib/
CUDADIR =../cuda/
LDIR =-L$(BOOST_LIB) -L$(CUDAPATH)/include/
LIBS = -lboost_thread -lcudart -lcuda

# Compiler options
CFLAGS= -c -m64 -O3 -Wall $(IDIR)
CUDAFLAGS= -c -m64 -O3 $(IDIR) -arch $(GPUARCH)
LDFLAGS= $(LIBS) $(LDIR) $(LIBS)

# Sources and Common clases sources
SOURCES= main.cpp
CSOURCES= $(CDIR)Histogram.cpp $(CDIR)JointProb.cpp $(CDIR)MutualInfo.cpp $(CDIR)ProbTable.cpp $(CDIR)RawData.cpp
CUDASOURCES= $(CUDADIR)histogram.cu $(CUDADIR)histogram64.cu $(CUDADIR)histogramJoint.cu

# Objects
OBJECTS=$(SOURCES:.cpp=.o)
COBJECTS=$(CSOURCES:.cpp=.o)
CUDAOBJECTS=$(CUDASOURCES:.cu=.o)

# Name of the executable
EXECUTABLE=fast-mrmr

.SUFFIXES: 

.SUFFIXES: .cu .cpp .o

all: $(SOURCES) $(CSOURCES) $(CUDASOURCES) $(EXECUTABLE)
$(EXECUTABLE): $(OBJECTS) $(COBJECTS) $(CUDAOBJECTS)
	$(NVCC) $(OBJECTS) $(COBJECTS) $(CUDAOBJECTS) $(LDFLAGS) -o $@

.cpp.o: $(SOURCES) $(CSOURCES)
	$(CC) $(CFLAGS) $< -o $@

.cu.o: $(CUDASOURCES)
	$(NVCC) $(CUDAFLAGS) $< -o $@

clean:
	rm *.o
	rm $(CDIR)*.o
	rm $(CUDADIR)*.o
	rm fast-mrmr
