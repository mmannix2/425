#include <unistd.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>

#define N 50000
#define CORES 50000

/* run the collatz conjecture and return the number of steps */
__global__ void collatz(unsigned int* step) {
    //Set x to the initial value of step
    unsigned int x = step[blockIdx.x];
    //Reset step to 0
    step[blockIdx.x] = 0;

    /* do the iterative process */
    while (x != 1) {
        if ((x % 2) == 0) {
            x = x / 2;
        } else {
            x = 3 * x + 1;
        }
        step[blockIdx.x]++;
    }
}

int main( ) {
    /* store the number of steps for each number up to N */
    unsigned int cpu_steps[N];
    unsigned int* gpu_steps;
    
    /* allocate space on the GPU */
    cudaMalloc((void**) &gpu_steps, N * sizeof(unsigned int));
    
    for(int i=0; i < N; i++) {
        cpu_steps[i] = i+1;
    }

    /* send gpu_steps to the GPU */
    cudaMemcpy(gpu_steps, cpu_steps, N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    /* run the collatz conjecture on all N items */
    collatz<<<CORES, 1>>>(gpu_steps);
    
    /* send gpu_steps back to the CPU */
    cudaMemcpy(cpu_steps, gpu_steps, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    /* free the memory on the GPU */
    cudaFree(gpu_steps);

    /* find the largest */
    unsigned int largest = cpu_steps[0], largest_i = 0;
    for (int i = 1; i < N; i++) {
        if (cpu_steps[i] > largest) {
            largest = cpu_steps[i];
            largest_i = i;
        }
    }

    /* report results */
    printf("The longest collatz chain up to %d is %d with %d steps.\n",
            N, largest_i + 1, largest);

    return 0;
}

