/*
Matt Mannix
CPSC 425
Lab2 Thread Timing

Compile with "make" and run "script.sh"

*/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define TOTAL_TOSSES 800000000

int THREADS = 1;
int TOSSES = TOTAL_TOSSES;

//The function that each thread runs
void* toss (void* int_ptr) {
    time_t startTime;
    time_t endTime;
    time(&startTime);
    
    int* seed = (int*) int_ptr;
    double* approx = malloc(sizeof(double));
    int hits = 0;
    double x; 
    double y;
    
    #ifdef DEBUG
    printf("Thread ID: %d\n", *seed);
    #endif

    //Toss the darts
    int i;
    for(i = 0; i < TOSSES; i++) {
        x = ((double) rand_r(seed) / (double) RAND_MAX) * 2.0 - 1.0;
        y = ((double) rand_r(seed) / (double) RAND_MAX) * 2.0 - 1.0;
        
        //DEBUG
        #ifdef DEBUG
        printf("(%.02f, %.02f) ", x, y);
        printf("\tDistance = %.02f. ", ((x*x + y*y)) );
        printf("\tHit = %d.\n", ((x*x + y*y) <= 1) );
        #endif

        hits += ((x*x + y*y) <= 1);
    }
    //DEBUG
    #ifdef DEBUG
    printf("hits: %d tosses: %d.\n", hits, TOSSES); 
    #endif

    *approx = (4.0 * ((double) hits / (double) TOSSES));
   
    printf("Execution time: %.03f.\n", difftime(time(&endTime), startTime));
    
    return (void*) approx;
}

//Main function, takes the number of cores as a command line argument
int main(int argc, char** argv) {
    double approx = 0.0;
    
    if(argc > 1) {
        //THREADS = (int) argv[1][0] - '0';
        THREADS = atoi(argv[1]);
        TOSSES = TOTAL_TOSSES / THREADS;
    }
    
    printf("Using %d threads and %d tosses per core.\n", THREADS, TOSSES);

    //Make some threads
    pthread_t threads[THREADS];
    int thread_ids[THREADS];
    
    //Spawn the threads
    int t;
    for(t = 0; t < THREADS; t++) {
        thread_ids[t] = t;
        pthread_create(&threads[t], NULL, toss, &thread_ids[t]);
    }
    
    //Join threads
    for(t = 0; t < THREADS; t++) {
        double* partial;
        pthread_join(threads[t], (void**) &partial);
        
        //DEBUG
        #ifdef DEBUG
        printf("Approximation %d: Pi = %1.04f.\n", t, *partial);
        #endif
        
        approx += *partial;
        free(partial);
    }
    
    //Average the values in approx
    approx /= THREADS;

    //Print output and end
    printf("Pi is roughly %1.06f.\n", approx);
    pthread_exit(NULL);
}

