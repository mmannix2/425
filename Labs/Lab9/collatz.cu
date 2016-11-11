#include <stdio.h>
#include <stdlib.h>

#define N 50000

/* run the collatz conjecture and return the number of steps */
__global__ unsigned int collatz(unsigned int x) {
    unsigned int step = 0;

    /* do the iterative process */
    while (x != 1) {
        if ((x % 2) == 0) {
            x = x / 2;
        } else {
            x = 3 * x + 1;
        }
        step++;
    }

    return step;
}

int main( ) {
    /* store the number of steps for each number up to N */
    unsigned int steps[N];

    /* run the collatz conjecture on all N items */
    unsigned int i;
    for (i = 0; i < N; i++) {
        steps[i] = collatz(i + 1);
    }

    /* find the largest */
    unsigned int largest = steps[0], largest_i = 0;
    for (i = 1; i < N; i++) {
        if (steps[i] > largest) {
            largest = steps[i];
            largest_i = i;
        }
    }

    /* report results */
    printf("The longest collatz chain up to %d is %d with %d steps.\n",
            N, largest_i + 1, largest);

    return 0;
}

