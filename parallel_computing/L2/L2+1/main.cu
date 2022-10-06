#include "cuda_runtime.h"      // CUDA runtime
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define N 80
#define M 80


struct Complex {
    float r;
    float i;
};

// Фуккция, которая выполняется на GPU
__global__ void Go(Complex *mass, Complex *rezalt, int n, int m){
    int tid = blockIdx.x; // Получить собственный индекс
    
    if (tid < n){
        rezalt[tid] = mass[tid*N];
        for(int i = 0; i < m; i++){
            if (sqrt(mass[tid*N + i].r * mass[tid*N + i].r +  mass[tid*N + i].i * mass[tid*N + i].i) > \
            sqrt(rezalt[tid].r * rezalt[tid].r +  rezalt[tid].i * rezalt[tid].i)){
                rezalt[tid] = mass[tid*N + i];
            }

        }
    }
}

void PrintMss(Complex mass[][N],int n, int m){
    printf("n = %d\n",n);
    printf("m = %d\n",m);
    printf("A = \n");

    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j<m; j++)
            printf("%fr %fi   ",mass[i][j].r,mass[i][j].i);
        printf("\n");
    }
    printf("\n");
}


int main(){
    int n;
    int m;
    FILE *f;
    Complex mass[N][M];
    Complex rezalt[N];
    float r, i;
    size_t pitch;

    f = fopen("data.txt","r");
    fscanf(f,"%d",&n);
    fscanf(f,"%d",&m);
    for(int j = 0; j < n; j++)
    {
        for(int k = 0; k < m; k++){
            fscanf(f,"%f",&mass[j][k].r);
            fscanf(f,"%f",&mass[j][k].i);
        }
    }
   
    fclose(f);
    printf("DATA\n");
    PrintMss(mass,n,m);

    Complex *dev_mass; // Адрес масива на GPU
	// Выделить память на GPU и сохранить её адрес в переменную
	cudaMalloc( (void**)&dev_mass, N * M * sizeof(Complex)); 
	// Копировать данные по адресу mass, размером N*sizeof(float) в адрес dev_mass. Копирование с устройства на GPU
	cudaMemcpy(dev_mass, mass, N * M * sizeof(Complex), cudaMemcpyHostToDevice);


    Complex *dev_rezalt; // Адрес масива на GPU
	// Выделить память на GPU и сохранить её адрес в переменную
	cudaMalloc( (void**)&dev_rezalt, N * sizeof(Complex)); 

    Go<<<n,1>>> (dev_mass,dev_rezalt,n,m);

    // Копировать данные по адресу dev_mass, размером N*sizeof(int) в адрес mass_GPU. Копирование с GPU на устройство
	cudaMemcpy(rezalt, dev_rezalt, N *sizeof(Complex), cudaMemcpyDeviceToHost);

    printf("REZALT\n");
    for(int j = 0; j<n; j++)
            printf("%fr %fi   ",rezalt[j].r,rezalt[j].i);

	return 0;
}