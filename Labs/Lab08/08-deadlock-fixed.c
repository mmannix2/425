#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>

/* function to compute the square root of a value */
#define ERROR 1e-15
double square_root(double value) {
    double hi = 1.0;
    double lo = value;
    
    while (fabs(hi - lo) > ERROR) {
        hi = (hi + lo) / 2.0;
        lo = value / hi;
    }

    return hi;
}

int main(int argc, char** argv) {
    int rank, size;

    /* initialize MPI */
    MPI_Init(&argc, &argv);

    /* get the rank (process id) and size (number of processes) */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* only allow 2 processes to run */
    if (size != 2) {
        if (rank == 0) {
            printf("Only run with two processes!\n");
        }
        MPI_Finalize( );
        return 0;
    }

    /* calculate different values in each process */
    double mine, other = 0.0;
    if (rank == 0) {
        mine = square_root(2.0);
    } else {
        mine = square_root(3.0);
    }

    /* exchange the two values the processes */
    /* I swapped the order of the Recv and Send calls in the else branch, so
     * one process is sending and the other is recieving. Previously, both
     * processes were deadlocked because the were both trying to recieve and
     * waiting on the other thread to send.
     */
    if (rank == 0) {
        MPI_Recv(&other, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(&mine, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    } else {
        MPI_Send(&mine, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
        
        MPI_Recv(&other, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* print them out */
    printf("Process %d has %lf and %lf.\n", rank, mine, other);

    /* quit MPI */
    MPI_Finalize( );
    return 0;
}

