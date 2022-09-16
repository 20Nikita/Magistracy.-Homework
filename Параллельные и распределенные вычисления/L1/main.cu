#include "cuda_runtime.h"      // CUDA runtime
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctime> 

#define N 512

// Фуккция, которая выполняется на GPU
__global__ void add( int *mass){
	int tid = blockIdx.x; // Получить собственный индекс
	if (tid < N){
		int t = mass[tid];
		mass[tid] = mass[N - tid - 1];
		mass[N - tid - 1] = t;
	}
}

int main(){
	system("chcp 65001"); // Для Русских букв в консоли
	// Инициализация переменных
	int mass[N];
	int mass_GPU[N];
	int mass_CPU[N];
	for (int i = 0; i < N; i++){
		mass[i] = i+1;
		mass_CPU[i] = i+1;
	}
	printf("***Исходный массив***\n");
	for (int i = 0; i < N; i++){
		printf("%i ",mass[i]);
	}
	printf("\n");

	int start_CPU, time_CPU;
	start_CPU = clock();

	//Обратный массив на CPU
	for (int i = 0; i < int(N/2); i++){
		int t = mass_CPU[i];
		mass_CPU[i] = mass_CPU[N - i - 1];
		mass_CPU[N - i - 1] = t;
	}
	time_CPU = clock() - start_CPU;
	printf("***Обратный массив на CPU за %iмс***\n", time_CPU);
	for (int i = 0; i < N; i++){
		printf("%i ",mass_CPU[i]);
	}
	printf("\n");




	int *dev_mass; // Адрес масива на GPU
	// Выделить память на GPU и сохранить её адрес в переменную
	cudaMalloc( (void**)&dev_mass, N * sizeof(int)); 
	// Копировать данные по адресу mass, размером N*sizeof(int) в адрес dev_mass. Копирование с устройства на GPU
	cudaMemcpy(dev_mass, mass, N * sizeof(int), cudaMemcpyHostToDevice);

	// Чото для времени GPU
    cudaEvent_t start_GPU, stop_GPU;
    float gpuTime = 0.0f;
	cudaEventCreate ( &start_GPU );
	cudaEventCreate ( &stop_GPU );
	cudaEventRecord ( start_GPU, 0 );  

		// Запуск функции. N/2 потоков
		add<<<int(N/2),1>>> (dev_mass);

	// Чото для времени GPU
	cudaEventRecord ( stop_GPU, 0 );
	cudaEventSynchronize ( stop_GPU );
	cudaEventElapsedTime ( &gpuTime, start_GPU, stop_GPU );

	// Копировать данные по адресу dev_mass, размером N*sizeof(int) в адрес mass_GPU. Копирование с GPU на устройство
	cudaMemcpy(mass_GPU, dev_mass, N * sizeof(int), cudaMemcpyDeviceToHost);

	printf("***Обратный массив на GPU за %.2fмс***\n", gpuTime);
	for (int i = 0; i < N; i++){
		printf("%i ",mass_GPU[i]);
	}
	printf("\n");
	// Отчистить память
	cudaFree(dev_mass);
	system("pause");
	return 0;
}