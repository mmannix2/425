#include <stdlib.h>
#include <pthread.h>
#include <stdio.h>

#define THREADS 2
#define NUM_ACCOUNTS 4

/* an array of account balances */
double accounts[NUM_ACCOUNTS];

/* one mutex for each account */
pthread_mutex_t account_locks[NUM_ACCOUNTS];

void* worker(void* fname) {
    /* open the file */
    FILE* f = fopen((char*) fname, "r");
    if (!f) {
        fprintf(stderr, "Error could not open file '%s'!\n", (char*) fname);
    }

    /* loop through all the transactions */
    while (1) {
        /* read one transaction from the file */
        int to, from;
        double amount;
        fscanf(f, "%d %d %lf", &to, &from, &amount);

        /* if we hit the end of file, break out of the loop */
        if (feof(f)) {
            break;
        }

        /* lock the from account */
        pthread_mutex_lock(&account_locks[from]);
        /* do the transaction */
        //printf("From -= %.02f. ", amount);
        accounts[from] -= amount;
        /* unlock the from account */
        pthread_mutex_unlock(&account_locks[from]);
        
        /* lock the to account */
        pthread_mutex_lock(&account_locks[to]);
        /* do the transaction */
        //printf("To += %.02f.\n", amount);
        accounts[to] += amount;
        /* unlock the to accounts */
        pthread_mutex_unlock(&account_locks[to]);
    }

    /* close the file and return */
    fclose(f);
    pthread_exit(NULL);
}

int main ( ) {
    /* an array of threads */
    pthread_t threads[THREADS];

    /* an array of the file names used for input */
    char fnames[THREADS][16];

    int i;
    /* start all accounts at 100.00, and init mutexes */
    for (i = 0; i < NUM_ACCOUNTS; i++) {
        accounts[i] = 100.00;
        pthread_mutex_init(&account_locks[i], NULL);
    }

    /* spawn all threads */
    for (i = 0; i < THREADS; i++) {
        sprintf(fnames[i], "file%d.txt", i);
        pthread_create(&threads[i], NULL, worker, fnames[i]);
    }

    /* join all threads */
    for (i = 0; i < THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    /* we're all done! */
    printf("All transactions completed!\n");
    pthread_exit(0);
}

