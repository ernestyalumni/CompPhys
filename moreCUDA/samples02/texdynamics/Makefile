# Ernest Yeung
# ernestyalumni@gmail.com
# 20161116
NVCC = nvcc
NVCC_FLAGS = -std=c++11

GPP = g++
GPP_FLAGS = -std=c++11

physobjects    = dev_R2grid.o R2grid.o 
commonobjects  = texObjCls2d.o

all: texdynamics2d.exe

texdynamics2d.exe: texdynamics2d.o $(physobjects) $(commonobjects)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ 
		
texdynamics2d.o: texdynamics2d.cu 
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

# commonlib
texObjCls2d.o: ./commonlib/texObjCls2d.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@
	

# physics
dev_R2grid.o: ./physlib/dev_R2grid.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

R2grid.o: ./physlib/R2grid.cpp
	$(GPP) $(GPP_FLAGS) -c $< -o $@
		
clean:
		rm -f *.o *.exe
