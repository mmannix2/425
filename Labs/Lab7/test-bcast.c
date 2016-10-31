#include <stdio.h>
#include <mpi.h>

#define N 500000

int main(int argc, char** argv) {
    int rank, size;

    /* initialize MPI */
    MPI_Init(&argc, &argv);

    /* get the rank (process id) and size (number of processes) */
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* have process 0 do many sends */
    int i;
    for (i = 0; i < N; i++) {
        MPI_Bcast(&i, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    /* quit MPI */
    MPI_Finalize( );
    return 0;
}

