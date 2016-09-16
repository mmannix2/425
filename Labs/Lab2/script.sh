#!/bin/sh

filename="Lab2"

touch $filename.txt
if [ -e $filename.txt ]
then
    cat /dev/null >| $filename.txt
fi

for i in 1 2 4 8 16 32 64;
do
    echo "Running with $i core(s)..."
    (time --format "Elapsed time: %e" ./$filename $i) 2>&1 | tee -a $filename.txt
    echo "\tdone."
done

