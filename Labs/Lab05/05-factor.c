/* program which finds the prime factorization of any number */
#include <stdio.h>
#include <math.h>

int is_prime(unsigned long long value) {
    unsigned long long i;

    if (value == 2) {
        return 1;
    }

    /* look for a number that divides it evenly */
    for (i = 2; i <= ceil(sqrt(value)); i++) {
        if ((value % i) == 0) {
            return 0;
        }
    }

    return 1;
}

void factorize(unsigned long long value) {
    /* copy of the value we can modify */
    unsigned long long temp = value, i;

    /* try all numbers up to half this value */
    for (i = 2; i <= ceil(value / 2.0); i++) {

        /* if the number is prime, consider it */
        if (is_prime(i)) {

            /* each prime factor may be in there multiple times, get them all */
            while ((temp % i) == 0) {
                printf("\t%llu\n", i);
                temp /= i;
            }
        }
    }
}

int main(int argc, char** argv) {
    /* get input */
    if (argc < 2) {
        printf("Pass a number!\n");
        return 0;
    }
    unsigned long long value;
    sscanf(argv[1], "%llu", &value);

    /* get factors, unless prime */
    if (is_prime(value)) {
        printf("This number is prime!\n");
    } else {
        printf("The factors are:\n");
        factorize(value);
    }

    return 0;
}

