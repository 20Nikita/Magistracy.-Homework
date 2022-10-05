#include "cuda_runtime.h"      // CUDA runtime
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#define N 80
// Фуккция, которая выполняется на GPU
__global__ void Go(float *mass,float *b,float *x,int n){
    for (int i = 1; i < n; i++){
        mass[i*N + i] = mass[i*N + i] - mass[i*N + i - 1] * mass[(i-1)*N + i] / mass[(i-1)*N + i - 1];
        b[i] = b[i] - b[i-1] * mass[(i)*N + i - 1] / mass[(i-1)*N + i - 1];
        mass[(i)*N + i-1] = 0;
    }
    x[n-1] = b[n-1] / mass[(n-1)*N + n - 1];
    for (int i = n - 2; i >= 0; i--){
        x[i] = (b[i] - x[i+1] * mass[(i)*N + i + 1]) / mass[i*N + i];
    }
}

void PrintMss(float mass[][N],float *b,int n){
    printf("n = %d\n",n);
    printf("A = \n");

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j<n; j++)
            printf("%f  ",mass[i][j]);
        printf("\n");
    }
    printf("\nb = ");
    for(int i = 0; i < n; i++)
        printf("%f  ",b[i]);
    printf("\n");
}


int main(){
    int n;
    FILE *f;
    float mass[N][N];
    float b[N];
    size_t pitch;

    f = fopen("data.txt","r");
    fscanf(f,"%d",&n);
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j<n; j++)
            fscanf(f,"%f",&mass[i][j]);
    }
    for(int i = 0; i < n; i++)
        fscanf(f,"%f",&b[i]);
    fclose(f);
    printf("DATA\n");
    PrintMss(mass,b,n);

    float *dev_mass; // Адрес масива на GPU
	// Выделить память на GPU и сохранить её адрес в переменную
	cudaMalloc( (void**)&dev_mass, N * N * sizeof(float)); 
	// Копировать данные по адресу mass, размером N*sizeof(float) в адрес dev_mass. Копирование с устройства на GPU
	cudaMemcpy(dev_mass, mass, N * N * sizeof(float), cudaMemcpyHostToDevice);

    float *dev_b; // Адрес масива на GPU
	// Выделить память на GPU и сохранить её адрес в переменную
	cudaMalloc( (void**)&dev_b, N * sizeof(float)); 
	// Копировать данные по адресу mass, размером N*sizeof(float) в адрес dev_mass. Копирование с устройства на GPU
	cudaMemcpy(dev_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    float *dev_x; // Адрес масива на GPU
	// Выделить память на GPU и сохранить её адрес в переменную
	cudaMalloc( (void**)&dev_x, N * sizeof(float)); 

    Go<<<1,1>>> (dev_mass,dev_b,dev_x,n);

    // Копировать данные по адресу dev_mass, размером N*sizeof(int) в адрес mass_GPU. Копирование с GPU на устройство
	cudaMemcpy(mass, dev_mass, N * N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(b, dev_b, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("REZALT\n");
    PrintMss(mass,b,n);

    cudaMemcpy(b, dev_x, N * sizeof(int), cudaMemcpyDeviceToHost);
    printf("\nx = ");
    for(int i = 0; i < n; i++)
        printf("%f  ",b[i]);
    printf("\n");
	return 0;
}