/* learrays.cu
 * Ernest Yeung 
 * ernestyalumni@gmail.com
 * demonstrate arrays, but in CUDA C/C++
 * */
#include <iostream>
#include "./common/errors.h"

__constant__ float dev_hds[3];

__constant__ float3 dev_cnus[4];



void set2DerivativeParameters(const float hd_i[3] )
{
	float unscaled_cnu[4] {  2.f / 3.f  , -1.f / 12.f ,
					0.f  , 0.f} ;
		
	float3 *cnus = new float3[4];

	for (int nu = 0 ; nu < 4; ++nu ) {
		cnus[nu].x = unscaled_cnu[nu]*(1.f/hd_i[0] ); 
		cnus[nu].y = unscaled_cnu[nu]*(1.f/hd_i[1] ); 
		cnus[nu].z = unscaled_cnu[nu]*(1.f/hd_i[2] ); 
	}
	
	for (int nu = 0 ; nu < 4; ++nu ) {
		std::cout<< " cnus values : nu : " << nu << " : .x " << cnus[nu].x << " : .y " << 
			cnus[nu].y << " : .z " << cnus[nu].z << std::endl; 
	}
	
	cudaMemcpyToSymbol( dev_cnus, cnus, sizeof(float3)*4, 0, cudaMemcpyHostToDevice) ; // offset from start is 0
	
	delete[] cnus;
	
	
			
}

__device__ float dev_dirder2(float stencil[2][2], float c_nus[4]) {
	int NU {2};
	float tempvalue {0.f};
		
	for (int nu = 0; nu < NU; ++nu ) {
		tempvalue += c_nus[nu]*( stencil[nu][1] - stencil[nu][0] );
	}
	return tempvalue;
}

//__global__ void testdiv(float3 dfloat3* dev_divres_in) {
__global__ void testdiv(float3* dev_divres_in) {
	
	// sanity check
	for (int nu = 0 ; nu < 4; ++nu ) {
//		std::cout << " cnus values : nu : " << nu << " : .x " << dev_cnus_in[nu].x << " : .y " << 
//			dev_cnus_in[nu].y << " : .z " << dev_cnus_in[nu].z << std::endl; 
		printf( " cnus values : nu : %d : .x %f : .y %f : .z %f \n ", nu, dev_cnus[nu].x, 
			dev_cnus[nu].y, dev_cnus[nu].z );

	}
	
	float stencilx[2][2] { { 1.2f, 1.6f }, { 12.1f, 16.2f } };
	float stencily[2][2] { { 0.32f, 0.056f }, { 32.1f, 5.12f } };
	float stencilz[2][2] { { 3.712f, 0.036f }, { 0.5f, 26.2f } };
	
	float c_nusx[4] { dev_cnus[0].x, dev_cnus[1].x, dev_cnus[2].x, dev_cnus[3].x } ;
	float c_nusy[4] { dev_cnus[0].y, dev_cnus[1].y, dev_cnus[2].y, dev_cnus[3].y } ;
	float c_nusz[4] { dev_cnus[0].z, dev_cnus[1].z, dev_cnus[2].z, dev_cnus[3].z } ;

/*
	std::cout << " c_nusx : " << c_nusx[0] << " " << c_nusx[1] << " " << c_nusx[2] << " " << 
			c_nusx[3] << std::endl;
	std::cout << " c_nusy : " << c_nusy[0] << " " << c_nusy[1] << " " << c_nusy[2] << " " << 
			c_nusy[3] << std::endl;
	std::cout << " c_nusz : " << c_nusz[0] << " " << c_nusz[1] << " " << c_nusz[2] << " " << 
			c_nusz[3] << std::endl;
*/

	printf( " c_nusx : %f  %f  %f  %f \n ", c_nusx[0], c_nusx[1], c_nusx[2], c_nusx[3] );
	printf( " c_nusy : %f  %f  %f  %f \n ", c_nusy[0], c_nusy[1], c_nusy[2], c_nusy[3] );
	printf( " c_nusz : %f  %f  %f  %f \n ", c_nusz[0], c_nusz[1], c_nusz[2], c_nusz[3] );

			
	float divresx { dev_dirder2( stencilx, c_nusx ) } ;
	float divresy { dev_dirder2( stencily, c_nusy ) } ;
	float divresz { dev_dirder2( stencilz, c_nusz ) } ;
	
//	std::cout << " divresx : " << divresx << std::endl;
//	std::cout << " divresy : " << divresy << std::endl;
//	std::cout << " divresz : " << divresz << std::endl;
	
	printf( " divresx : %f \n " , divresx ) ;
	printf( " divresy : %f \n " , divresy ) ;
	printf( " divresz : %f \n " , divresz ) ;

	dev_divres_in->x = divresx;
	dev_divres_in->y = divresy;
	dev_divres_in->z = divresz;


}

__global__ void sanitycheck_assign( float3 *dev_result_in ) {
	dev_result_in->x = 1.f; 
	dev_result_in->y = 2.f; 
	dev_result_in->z = 3.f; 

	printf( " dev_result_in->x : %f \n " , dev_result_in->x ) ;
	printf( " dev_result_in->y : %f \n " , dev_result_in->y ) ;
	printf( " dev_result_in->z : %f \n " , dev_result_in->z ) ;

}

__global__ void sanitycheck_const() {
	for (int nu = 0; nu < 4 ; ++nu ) {
		printf( " dev_cnus for nu : %d  : .x : %f ,  .y : %f  .z : %f  \n " , nu, dev_cnus[nu].x ,
			dev_cnus[nu].y , dev_cnus[nu].z );

		
	}

}

// sanity check const2 doesn't work can't use dev_cnus as argument from host
/*
__global__ void sanitycheck_const2(float3 dev_cnus_in[4]) {
	for (int nu = 0; nu < 4 ; ++nu ) {
		printf( " dev_cnus for nu : %d  : .x : %f ,  .y : %f  .z : %f  \n " , nu, dev_cnus_in[nu].x ,
			dev_cnus_in[nu].y , dev_cnus_in[nu].z );

		
	}

}
*/


int main() {
	const float hds[3] { 0.1, 0.01, 0.001 };
	
	std::cout << " These are values for hds : " << hds[0] << " " << hds[1] << " " << hds[2] << std::endl;
	
	cudaMemcpyToSymbol( dev_hds, hds, sizeof(float)*3,0,cudaMemcpyHostToDevice) ;
	
	set2DerivativeParameters( hds );
	

// cf. http://stackoverflow.com/questions/24460507/cuda-invalid-argument-when-trying-to-copy-struct-to-devices-memory-cudamemcpy
// what DID NOT work: float3* divresult did not work; Segmentation Fault, memory address wasn't found
	float3 divresult;
	float3* dev_divresult;

	HANDLE_ERROR(
		cudaMalloc( (void**)&dev_divresult , sizeof(float3) ) );

// sanity check

	sanitycheck_assign<<<1,1>>>(dev_divresult) ;

	sanitycheck_const<<<1,1>>>() ;

// sanitycheck_const2 doesn't work
//	sanitycheck_const2<<<1,1>>>(dev_cnus) ;


	testdiv<<<1,1>>>(dev_divresult);
	
	HANDLE_ERROR(
		cudaMemcpy( &divresult, dev_divresult, sizeof(float3), cudaMemcpyDeviceToHost) ); 

// what DID NOT work; cudaMemcpy( divresult, ... ) when float3* divresult

	
	std::cout << " These are values for divresult, which was cudaMemcpy'ed from dev_divresult : .x " 
		<< divresult.x << " : .y " << divresult.y << " : .z " << divresult.z << std::endl;
	
//	std::cout << divresult->x << std::endl;
//	std::cout << (*divresult).x << std::endl;
	
	
	HANDLE_ERROR( 
		cudaFree( dev_divresult) );
}
