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
  if (rank == 0) {
    int i, j;
    for (i = 0; i < N; i++) {
      for (j = 1; j < size; j++) {
        MPI_Send(&i, 1, MPI_INT, j, 0, MPI_COMM_WORLD);
      }
    }
  }

  /* have the rest receive that many values */
  else {
    int i;
    for (i = 0; i < N; i++) {
      int value;
      MPI_Recv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  /* quit MPI */
  MPI_Finalize( );
  return 0;
}

