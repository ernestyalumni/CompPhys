# Ernest Yeung
# ernestyalumni@gmail.com
# 20170112
NVCC = nvcc
NVCC_FLAGS = -std=c++11

GPP = g++
GPP_FLAGS = -std=c++11

#physobjects    = dev_R2grid.o R2grid.o boundary.o u_p.o
physobjects    = R2grid.o dev_R2grid.o init.o PDE.o

all: main.exe

main.exe: main.o $(physobjects) 
#$(commonobjects) $(gfxobjects)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(GFX_FLAGS)
		
main.o: main.cu 
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

	

# physics
dev_R2grid.o: ./physlib/dev_R2grid.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

R2grid.o: ./physlib/R2grid.cpp
#	$(GPP) $(GPP_FLAGS) -c $< -o $@  # I need float2 and so I had to use the nvcc compiler
	$(NVCC) -x cu $(NVCC_FLAGS) -dc $< -o $@  

#boundary.o: ./physlib/boundary.cu  
# ./physlib/dev_R2grid.cu
#	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

#u_p.o: ./physlib/u_p.cu
#	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

PDE.o: ./physlib/PDE.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@


init.o: ./physlib/init.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@


		
clean:
		rm -f *.o *.exe
