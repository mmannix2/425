#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>

#define THREADS 4
#define TOSSES 50000000

/* the function called for each thread */
void* estimate_pi(void* idp) {
    int number_in_circle = 0, toss;
    int tosses = TOSSES;
    double rand_max = (double) RAND_MAX;

    /* set the seed to our thread id */
    unsigned int* seed = idp;

    /* loop through each toss */
    for (toss = 0; toss < tosses; toss++) {
        double x = ((double) rand_r(seed) / rand_max) * 2.0 - 1.0;
        double y = ((double) rand_r(seed) / rand_max) * 2.0 - 1.0;
        double dist_squared = x*x + y*y;

        if (dist_squared <= 1) {
            number_in_circle++;
        }
    }

    /* calculate this thread's pi estimate */
    double* pi_estimate = malloc(sizeof(double));
    *pi_estimate = (4 * number_in_circle) / (double) tosses;

    pthread_exit(pi_estimate);
}

int main ( ) {
    /* an array of threads */
    pthread_t threads[THREADS];
    int ids[THREADS];
    int i;

    /* spawn all threads */
    for (i = 0; i < THREADS; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, estimate_pi, &ids[i]);
    }

    /* join all threads collecting answer */
    double total = 0;
    for (i = 0; i < THREADS; i++) {
        double* answer;
        pthread_join(threads[i], (void**) &answer);
        total += *answer;
        free(answer);
    }

    /* print result */
    double final = total / THREADS;
    printf("Pi is roughly %lf\n", final); 
    pthread_exit(0);
}

