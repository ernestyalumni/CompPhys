
/*
Simulation of flow inside a 2D square cavity
using the lattice Boltzmann method (LBM)

Written by:       Abhijit Joshi (joshi1974@gmail.com)

Build instructions: nvcc -arch=sm_13 gpu_lbm.cu -o lbmGPU.x

Run instructions: ./lbmGPU.x
*/

#include<iostream>

// problem parameters
const int N = 512;            // number of node points along X and Y (cavity length in lattice units)
const int TIME_STEPS = 3;  // number of time steps for which the simulation is run; originally 1000

const int NDIR = 9;           // number of discrete velocity directions used in the D2Q9 model

const double DENSITY = 2.7;          // fluid density in lattice units
const double LID_VELOCITY = 0.05;    // lid velocity in lattice units
const double REYNOLDS_NUMBER = 100;  // Re = 

__global__ void initialize(double *_ex, double *_ey, double *_alpha, int *_ant, double *_rh, double *_ux, double *_uy, double *_f, double *_feq, double *_f_new)
{
    // these could be initialized directly inside the GPU kernel and moved out of the CPU part...
    _ex[0] =  0.0;   _ey[0] =  0.0;   _alpha[0] = 4.0 /  9.0;
    _ex[1] =  1.0;   _ey[1] =  0.0;   _alpha[1] = 1.0 /  9.0;
    _ex[2] =  0.0;   _ey[2] =  1.0;   _alpha[2] = 1.0 /  9.0;
    _ex[3] = -1.0;   _ey[3] =  0.0;   _alpha[3] = 1.0 /  9.0;
    _ex[4] =  0.0;   _ey[4] = -1.0;   _alpha[4] = 1.0 /  9.0;
    _ex[5] =  1.0;   _ey[5] =  1.0;   _alpha[5] = 1.0 / 36.0;
    _ex[6] = -1.0;   _ey[6] =  1.0;   _alpha[6] = 1.0 / 36.0;
    _ex[7] = -1.0;   _ey[7] = -1.0;   _alpha[7] = 1.0 / 36.0;
    _ex[8] =  1.0;   _ey[8] = -1.0;   _alpha[8] = 1.0 / 36.0;

    // these could be initialized directly inside the GPU kernel and moved out of the CPU part...
    _ant[0] = 0;      //      6        2        5
    _ant[1] = 3;      //               ^       
    _ant[2] = 4;      //               |
    _ant[3] = 1;      //               |  
    _ant[4] = 2;      //      3 <----- 0 -----> 1
    _ant[5] = 7;      //               |
    _ant[6] = 8;      //               |
    _ant[7] = 5;      //               v
    _ant[8] = 6;      //      7        4        8

    // compute the "i" and "j" location and the "dir"
    // handled by this thread

    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;

    // initialize density and velocity fields inside the cavity
    _rh[i+N*j] = DENSITY;
    _ux[i+N*j] = 0;
    _uy[i+N*j] = 0;

    if(j==N-1) _ux[i+N*(N-1)] = LID_VELOCITY;

    // assign initial values for distribution functions
    int ixy = i+N*j;
    for(int dir=0;dir<NDIR;dir++) {
        int index = i+j*N+dir*N*N;
        double edotu = _ex[dir]*_ux[ixy] + _ey[dir]*_uy[ixy];
        double udotu = _ux[ixy]*_ux[ixy] + _uy[ixy]*_uy[ixy];
        _feq[index] = _rh[ixy] * _alpha[dir] * (1 + 3*edotu + 4.5*edotu*edotu - 1.5*udotu);
        _f[index] = _feq[index];
        _f_new[index] = _feq[index];
    }
}

__global__ void timeIntegration(double *_ex, double *_ey, double *_alpha, int *_ant, double *_rh, double *_ux, double *_uy, double *_f, double *_feq, double *_f_new)
{
    // calculate fluid viscosity based on the Reynolds number
    double kinematicViscosity = LID_VELOCITY * (double)N / REYNOLDS_NUMBER;

    // calculate relaxation time tau
    double tau =  0.5 + 3.0 * kinematicViscosity;

    // compute the "i" and "j" location and the "dir"
    // handled by this thread

    int i = blockIdx.x * blockDim.x + threadIdx.x ;
    int j = blockIdx.y * blockDim.y + threadIdx.y ;

    {
        // collision
        if((i>0) && (i<N-1) && (j>0) && (j<N-1)) {
                int ixy = i+N*j;
                for(int dir=0;dir<NDIR;dir++) {
                    int index = i+j*N+dir*N*N;
                    double edotu = _ex[dir]*_ux[ixy] + _ey[dir]*_uy[ixy];
                    double udotu = _ux[ixy]*_ux[ixy] + _uy[ixy]*_uy[ixy];
                    _feq[index] = _rh[ixy] * _alpha[dir] * (1 + 3*edotu + 4.5*edotu*edotu - 1.5*udotu);
                }
        }

        // streaming from interior node points
        if((i>0) && (i<N-1) && (j>0) && (j<N-1)) {
                for(int dir=0;dir<NDIR;dir++) {

                    int index = i+j*N+dir*N*N;   // (i,j,dir)
                    int index_new = (i+_ex[dir]) + (j+_ey[dir])*N + dir*N*N;
                    int index_ant = i + j*N + _ant[dir]*N*N;

                    // post-collision distribution at (i,j) along "dir"
                    double f_plus = _f[index] - (_f[index] - _feq[index])/tau;

                    if((i+_ex[dir]==0) || (i+_ex[dir]==N-1) || (j+_ey[dir]==0) || (j+_ey[dir]==N-1)) {
                        // bounce back
                        int ixy = i+_ex[dir] + N*(j+_ey[dir]);
                        double ubdote = _ux[ixy]*_ex[dir] + _uy[ixy]*_ey[dir];
                        _f_new[index_ant] = f_plus - 6.0 * DENSITY * _alpha[dir] * ubdote;
                    }
                    else {
                        // stream to neighbor
                        _f_new[index_new] = f_plus;
                    }
                }
        }

        // push f_new into f
        if((i>0) && (i<N-1) && (j>0) && (j<N-1)) {
                for(int dir=0;dir<NDIR;dir++) {
                    int index = i+j*N+dir*N*N;   // (i,j,dir)
                    _f[index] = _f_new[index];
                }
        }

        // update density at interior nodes
        if((i>0) && (i<N-1) && (j>0) && (j<N-1)) {
                double rho=0;
                for(int dir=0;dir<NDIR;dir++) {
                    int index = i+j*N+dir*N*N;
                    rho+=_f_new[index];
                }
                _rh[i+N*j] = rho;
        }

        // update velocity at interior nodes
        if((i>0) && (i<N-1) && (j>0) && (j<N-1)) {
                double velx=0;
                double vely=0;
                for(int dir=0;dir<NDIR;dir++) {
                    int index = i+j*N+dir*N*N;
                    velx+=_f_new[index]*_ex[dir];
                    vely+=_f_new[index]*_ey[dir];
                }
                _ux[i+N*j] = velx/_rh[i+N*j];
                _uy[i+N*j] = vely/_rh[i+N*j];
        }
    }

}

int main(int argc, char *argv[])
{
    // the base vectors and associated weight coefficients (GPU)
    double *_ex, *_ey, *_alpha;  // pointers to device (GPU) memory
    cudaMalloc((void **)&_ex,NDIR*sizeof(double));
    cudaMalloc((void **)&_ey,NDIR*sizeof(double));
    cudaMalloc((void **)&_alpha,NDIR*sizeof(double));

    // ant vector (GPU)
    int *_ant;  // gpu memory
    cudaMalloc((void **)&_ant,NDIR*sizeof(int));

    // allocate memory on the GPU
    double *_f, *_feq, *_f_new;
    cudaMalloc((void **)&_f,N*N*NDIR*sizeof(double));
    cudaMalloc((void **)&_feq,N*N*NDIR*sizeof(double));
    cudaMalloc((void **)&_f_new,N*N*NDIR*sizeof(double));

    double *_rh, *_ux, *_uy;
    cudaMalloc((void **)&_rh,N*N*sizeof(double));
    cudaMalloc((void **)&_ux,N*N*sizeof(double));
    cudaMalloc((void **)&_uy,N*N*sizeof(double));

    // assign a 3D distribution of CUDA "threads" within each CUDA "block"    
    int threadsAlongX=16, threadsAlongY=16;
    dim3 dimBlock(threadsAlongX, threadsAlongY, 1);

    // calculate number of blocks along X and Y in a 2D CUDA "grid"
    dim3 dimGrid( ceil(float(N)/float(dimBlock.x)), ceil(float(N)/float(dimBlock.y)), 1 );

    initialize<<<dimGrid,dimBlock>>>(_ex, _ey, _alpha, _ant, _rh, _ux, _uy, _f, _feq, _f_new);

    // time integration
    int time=0;
    while(time<TIME_STEPS) {

        time++;

        timeIntegration<<<dimGrid,dimBlock >>>(_ex, _ey, _alpha, _ant, _rh, _ux, _uy, _f, _feq, _f_new);

    }


	// sanity check
	
	// initialize density and velocity fields inside the cavity on host CPU
	double *ux = new double[ N * N ] ;
	double *uy = new double[ N * N ] ;
	double *rh = new double[ N * N ] ;

	cudaMemcpy( rh, _rh, sizeof(double) * N * N, cudaMemcpyDeviceToHost) ;

	std::cout << "\n In the middle: " << std::endl ;
	std::cout << " _rh : " << std::endl ;
	for (int j = N/2 - 10 ; j < N/2 + 10; j++) {
		for (int i = N/2 - 10; i < N/2 + 10; i++) {
			std::cout << rh[ i + N * j ] << " " ; 
		}
		std::cout << std::endl ;
	}

	cudaMemcpy( ux, _ux, sizeof(double) * N * N, cudaMemcpyDeviceToHost) ;

	std::cout << " _ux : " << std::endl ;
	for (int j = N/2 - 10 ; j < N/2 + 10; j++) {
		for (int i = N/2 - 10; i < N/2 + 10; i++) {
			std::cout << ux[ i + N * j ] << " " ; 
		}
		std::cout << std::endl ;
	}

	cudaMemcpy( uy, _uy, sizeof(double) * N * N, cudaMemcpyDeviceToHost) ;

	std::cout << " _uy : " << std::endl ;
	for (int j = N/2 - 10 ; j < N/2 + 10; j++) {
		for (int i = N/2 - 10; i < N/2 + 10; i++) {
			std::cout << uy[ i + N * j ] << " " ; 
		}
		std::cout << std::endl ;
	}

	std::cout << "\n on the corner : " << std::endl ;
	std::cout << " _rh : " << std::endl ;
	for (int j = N - 20 ; j < N; j++) {
		for (int i = N - 20; i < N ; i++) {
			std::cout << rh[ i + N * j ] << " " ; 
		}
		std::cout << std::endl ;
	}

	cudaMemcpy( ux, _ux, sizeof(double) * N * N, cudaMemcpyDeviceToHost) ;

	std::cout << " _ux : " << std::endl ;
	for (int j = N - 20 ; j < N ; j++) {
		for (int i = N - 20; i < N ; i++) {
			std::cout << ux[ i + N * j ] << " " ; 
		}
		std::cout << std::endl ;
	}

	cudaMemcpy( uy, _uy, sizeof(double) * N * N, cudaMemcpyDeviceToHost) ;

	std::cout << " _uy : " << std::endl ;
	for (int j = N - 20 ; j < N ; j++) {
		for (int i = N  - 20; i < N  ; i++) {
			std::cout << uy[ i + N * j ] << " " ; 
		}
		std::cout << std::endl ;
	}
	// END sanity check




	// free host CPU memory
	delete[] ux;
	delete[] uy;
	delete[] rh;

	// free device GPU memory
	cudaFree( _ex ) ; 
	cudaFree( _ey ); 
	cudaFree( _alpha );
	cudaFree( _ant );


	cudaFree( _f ) ; 
	cudaFree( _feq ); 
	cudaFree( _f_new );
	
	cudaFree( _rh ) ; 
	cudaFree( _ux ); 
	cudaFree( _uy );
	


    return 0;
}
