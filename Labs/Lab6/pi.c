#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define TOSSES 100000000

/* estimate Pi using a Monte Carlo method */
double estimate_pi( ) {
    int number_in_circle = 0, toss;
    
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
    int rank;
    double pi; 
    
    /* initialize MPI */
    MPI_Init(&argc, &argv);
    
    /* get the rank */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if(rank == 0) {
        /* print result */
        //pi = estimate_pi( );
        printf("Pi is roughly %lf\n", pi); 
    }
    else {
        printf("Rank: %d\n", rank);
    }

    /* quit MPI */
    MPI_Finalize();

    return 0;
}

