# Ernest Yeung
# ernestyalumni@gmail.com
# 20180116
NVCC = nvcc
NVCC_FLAGS = -std=c++14

GPP = g++
GPP_FLAGS = -std=c++11

GFX_FLAGS = -lglut -lGL -lGLU 

physobjects    = grid2d.o sysparam.o metropolis.o 
# boundary.o
utilobjects    = XORMRGgens.o gridsetup.o 
FileIOobjects  = output.o 

all: main

main: main.o $(physobjects) $(utilobjects) $(FileIOobjects) 
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ 
		
main.o: main.cu 
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

	
# utilities (random number generators), common

XORMRGgens.o: ./dynam/XORMRGgens.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

gridsetup.o: ./common/gridsetup.cu
	$(NVCC) $(NVCC_FLAGS) -dc ./common/gridsetup.cu
#	$(NVCC) $(NVCC_FLAGS) -dc S< -o S@

# physics
grid2d.o: ./grid2d/grid2d.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

sysparam.o: ./grid2d/sysparam.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@
	
#metropolis.o: ./dynam/XORMRGgens.cu ./dynam/metropolis.cu ./grid2d/sysparam.cu ./grid2d/grid2d.cu
#	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

metropolis.o: ./dynam/metropolis.cu ./dynam/XORMRGgens.h ./grid2d/sysparam.h ./grid2d/grid2d.h
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@


#boundary.o: ./boundary/boundary.cu  
#	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@


# File IO

output.o: ./FileIO/output.cu ./grid2d/sysparam.h
	$(NVCC) $(NVCC_FLAGS) -dc ./FileIO/output.cu
#	$(NVCC) $(NVCC_FLAGS) -dc $< -o S@


		
clean:
		rm -f *.o *.exe
