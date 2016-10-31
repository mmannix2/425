#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define TOSSES 100000000

/* estimate Pi using a Monte Carlo method */
double estimate_pi(int seed) {
    int number_in_circle = 0, toss;
    srand(seed);

    /* loop through each toss */
    for (toss = 0; toss < TOSSES; toss++) {
        double x = ((double) rand( ) / (double) RAND_MAX) * 2.0 - 1.0;
        double y = ((double) rand( ) / (double) RAND_MAX) * 2.0 - 1.0;
        double dist_squared = x*x + y*y;

        if (dist_squared <= 1) {
        number_in_circle++;
        }
    } 

    /* calculate this thread's pi estimate */
    double pi_estimate = (4 * number_in_circle) / (double) TOSSES;
    return pi_estimate;
}

int main (int argc, char** argv) {
    int rank, size;
    double pi = 0.0;
    
    /* initialize MPI */
    MPI_Init(&argc, &argv);
    
    /* get the rank and size */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /* extimate pi */
    pi = estimate_pi(rank);
    
    /* Master process */
    if(rank == 0) {
        double partial = 0.0;
        /* recieve results from other processes */
        for(int i=1; i < size; i++) {
            MPI_Status status;
            MPI_Recv(&partial, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            pi += partial;
        }

        /* average results */
        pi /= size;

        /* print result */
        printf("Pi is roughly %lf\n", pi); 
    }
    /* Slave processes */
    else {
        MPI_Send(&pi, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    /* quit MPI */
    MPI_Finalize();

    return 0;
}
