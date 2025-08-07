# Makefile for OpenMP programs on macOS

CXX = g++
CXXFLAGS = -std=c++11 -Xpreprocessor -fopenmp -O2
INCLUDES = -I/opt/homebrew/opt/libomp/include
LIBS = -L/opt/homebrew/opt/libomp/lib -lomp

# Targets
TARGETS = numerical-integration blocked-matrix-multiplication

all: $(TARGETS)

numerical-integration: numerical-integration.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(LIBS) -o $@

blocked-matrix-multiplication: blocked-matrix-multiplication.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< $(LIBS) -o $@

clean:
	rm -f $(TARGETS)

.PHONY: all clean
