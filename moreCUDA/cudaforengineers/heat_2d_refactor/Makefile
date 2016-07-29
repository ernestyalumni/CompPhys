LDFLAGS += -L/usr/local/cuda/samples/common/lib/linux/x86_64
LDFLAGS += -lglut -lGL -lGLU

NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -g -G -Xcompiler "-Wall -Wno-deprecated-declarations"
INC = -I/usr/local/cuda/samples/common/inc

all: main.exe

main.exe: main.o finitediff.o tex_anim2d.o heat_2d.o dev_R2grid.o R2grid.o
		$(NVCC) -std=c++11 $^ -o $@ $(LDFLAGS)
		
main.o: main.cu 
		$(NVCC) -std=c++11 $(NVCC_FLAGS) $(INC) -dc $< -o $@

finitediff.o: ./commonlib/finitediff.cu
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -dc $< -o $@
	

tex_anim2d.o: ./commonlib/tex_anim2d.cu 
		$(NVCC) -std=c++11 $(NVCC_FLAGS) $(LDFLAGS) -dc $< -o $@

# physics

heat_2d.o: ./physlib/heat_2d.cu 
		$(NVCC) -std=c++11 $(NVCC_FLAGS) -dc $< -o $@

dev_R2grid.o: ./physlib/dev_R2grid.cu
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -dc $< -o $@

R2grid.o: ./physlib/R2grid.cpp
	$(NVCC) -std=c++11 $(NVCC_FLAGS) -dc $< -o $@

	
		
clean:
		rm -f *.o *.exe
