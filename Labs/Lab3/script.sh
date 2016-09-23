#!/bin/bash

PROGRAM=$1

for i in `seq 1 100`;
do
    echo "*** $PROGRAM RUN #$i ***"
    ./$PROGRAM
done
