#include "cuda_runtime.h"      // CUDA runtime
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
// Фуккция, которая выполняется на GPU
__global__ void Go(float x,float *rez,int a,int m,int c,int n){
    *rez = (float)((int)(a*x + c) % m) / (float)m;
}



int main(){
    system("chcp 65001"); // Для Русских букв в консоли
    float x;
    int a = 593456;
    int m = 5936;
    int c = 532673;
    int n;
    

    printf("ВВедите начальное число ");
    scanf("%f", &x); // ввод  переменной a с клавиатуры
    printf("\n");
    printf("Сколько случайных чисел необходимо сгененрровать? ");
    scanf("%d", &n); // ввод  переменной a с клавиатуры
    printf("\n");

    float *dev_rez; // Адрес масива на GPU
	// Выделить память на GPU и сохранить её адрес в переменную
	cudaMalloc((void**)&dev_rez, sizeof(float));

    for(int i = 0; i < n; i++){

        Go<<<1,1>>> (x,dev_rez,a,m,c,n);

        // Копировать данные по адресу dev_mass, размером N*sizeof(int) в адрес mass_GPU. Копирование с GPU на устройство
        cudaMemcpy(&x, dev_rez, sizeof(float), cudaMemcpyDeviceToHost);

        printf("Случайное число №%d: %f \n",i,x);
    }
	return 0;
}