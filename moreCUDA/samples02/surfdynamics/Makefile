# Ernest Yeung
# ernestyalumni@gmail.com
# 20161116
NVCC = nvcc
NVCC_FLAGS = -std=c++11

GPP = g++
GPP_FLAGS = -std=c++11

physobjects    = dev_R2grid.o R2grid.o 
commonobjects  = surfObj2d.o

all: surfdynamics2d.exe

surfdynamics2d.exe: surfdynamics2d.o $(physobjects) $(commonobjects)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ 
		
surfdynamics2d.o: surfdynamics2d.cu 
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

# commonlib
surfObj2d.o: ./commonlib/surfObj2d.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@
	

# physics
dev_R2grid.o: ./physlib/dev_R2grid.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

R2grid.o: ./physlib/R2grid.cpp
#	$(GPP) $(GPP_FLAGS) -c $< -o $@  # I need float2 and so I had to use the nvcc compiler
	$(NVCC) -x cu $(NVCC_FLAGS) -dc $< -o $@  
		
clean:
		rm -f *.o *.exe
