# Ernest Yeung
# ernestyalumni@gmail.com
# 20170112
NVCC = nvcc
NVCC_FLAGS = -std=c++11

GPP = g++
GPP_FLAGS = -std=c++11

GFX_FLAGS = -lglut -lGL -lGLU 

physobjects    = R2grid.o dev_R2grid.o init.o PDE.o
gfxobjects     = tex_anim2d.o


all: main.exe

main.exe: main.o $(physobjects) $(gfxobjects)
	$(NVCC) $(NVCC_FLAGS) $^ -o $@ $(GFX_FLAGS)
		
main.o: main.cu 
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

	

# physics
dev_R2grid.o: ./physlib/dev_R2grid.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@

R2grid.o: ./physlib/R2grid.cpp
#	$(GPP) $(GPP_FLAGS) -c $< -o $@  # I need float2 and so I had to use the nvcc compiler
	$(NVCC) -x cu $(NVCC_FLAGS) -dc $< -o $@  


PDE.o: ./physlib/PDE.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@


init.o: ./physlib/init.cu
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $@


# graphics (gfx)
tex_anim2d.o: ./commonlib/tex_anim2d.cu
	$(NVCC) $(NVCC_FLAGS) $(GFX_FLAGS) -dc $< -o $@


		
clean:
		rm -f *.o *.exe
