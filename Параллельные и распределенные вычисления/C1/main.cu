#include "cuda_runtime.h"      // CUDA runtime
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctime> 


// Фуккция, которая выполняется на GPU
__global__ void add(){
	printf("***CUDA***\n");
}

int main(){

	printf("***CPU***\n");
	add<<<2,2>>> ();
	return 0;
}