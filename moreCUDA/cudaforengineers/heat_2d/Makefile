LDFLAGS += -L/usr/local/cuda/samples/common/lib/linux/x86_64
LDFLAGS += -lglut -lGL -lGLU

NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-declarations"
INC = -I/usr/local/cuda/samples/common/inc

all: main.exe

main.exe: main.o heat_2d.o
		$(NVCC) $^ -o $@ $(LDFLAGS)
		
main.o: main.cpp heat_2d.h interactions.h
		$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@
		
heat_2d.o: heat_2d.cu heat_2d.h
		$(NVCC) $(NVCC_FLAGS) -c $< -o $@
		
clean:
		rm -f *.o *.exe
