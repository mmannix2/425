#!/bin/bash
BROKEN="04-sharing"
FIXED="$BROKEN-fixed"

make

echo $BROKEN
time ./$BROKEN
echo $FIXED
time ./$FIXED
