# Compiler definitions
CC = g++
NVCC = nvcc

CFLAGS = -O3 -finline-functions -ffast-math -fomit-frame-pointer -funroll-loops
INCPATH = -I. -I.

# Targets and files
CPU_OBJ = main.o
GPU_OBJ = exponentialIntegral.o
OBJ = $(CPU_OBJ) $(GPU_OBJ)

EXEC = exponentialIntegral.out

all: $(EXEC)

$(EXEC): $(OBJ)
	$(NVCC) -o $@ $^

# Compile .cpp files with g++
%.o: %.cpp Makefile
	$(CC) $(CFLAGS) -c $(INCPATH) $< -o $@

# Compile .cu files with nvcc
%.o: %.cu Makefile
	$(NVCC) -c $(INCPATH) $< -o $@

install:

clean:
	rm -f *.o $(EXEC)
