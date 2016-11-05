# Ernest Yeung
# ernestyalumni@gmail.com
# 20161104
NVCC = nvcc
NVCC_FLAGS = -std=c++11

methodobjects  = scans.o

all: main.exe

main.exe: main.o $(methodobjects)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@
	
main.o : main.cu $(methodobjects)
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

# Note to budding programmer: for some reason, CUDA C/C++ compiles C++ header files only as 
# separate objects; otherwise I obtain errors such as the following:
# main.o: In function `main':
# tmpxft_00004e72_00000000-4_main.cudafe1.cpp:(.text+0x19b): undefined reference to `Blelloch_scan_kernelLauncher(float*, float*, int, int)'
# collect2: error: ld returned 1 exit status
# Makefile:10: recipe for target 'main.exe' failed
# make: *** [main.exe] Error 1

# methods 

scans.o : ./methods/scans.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@



clean:
	rm -f *.o *.exe

