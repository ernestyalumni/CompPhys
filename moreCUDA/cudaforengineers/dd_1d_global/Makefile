NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler -Wall

all: main.exe

main.exe: main.o dd_1d_global.o
		$(NVCC) $^ -o $@
		
main.o: main.cpp dd_1d_global.h
		$(NVCC) $(NVCC_FLAGS) -c $< -o $@
		
dd_1d_global.o: dd_1d_global.cu dd_1d_global.h
		$(NVCC) $(NVCC_FLAGS) -c $< -o $@
	
clean:
	rm -f *.o *.exe
