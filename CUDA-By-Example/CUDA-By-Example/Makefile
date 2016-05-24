NVCCFLAGS 	:= -g --use_fast_math
LIBS		:= -Xlinker -framework,GLUT -Xlinker -framework,OpenGL
SRCS 		= $(wildcard *.cu)
BINS 		= $(patsubst %.cu,%,$(SRCS))

all: $(BINS)
		
%: %.cu
	nvcc $(NVCCFLAGS) $(LIBS) $< -o ./bin/$@

clean:
	rm ./bin/*
