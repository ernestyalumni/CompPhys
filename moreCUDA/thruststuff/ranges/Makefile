# Ernest Yeung
# ernestyalumni@gmail.com
# 20170109
NVCC = nvcc
NVCC_FLAGS = -std=c++11

GPP = g++
GPP_FLAGS = -std=c++11

#rangeobjects    = ranges.o

all: main.exe

#main.exe: main.o  $(rangeobjects) 
main.exe: main.o   
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ 

main.o: main.cu 
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

#ranges.o: ./ranges/ranges.cu
#	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

#ranges.o: ./ranges/ranges.cpp
#	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@


clean:
		rm -f *.o *.exe
