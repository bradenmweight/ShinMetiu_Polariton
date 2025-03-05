#!/bin/bash

for N in 10 25 50 75 100; do
    for A0 in 0.00 0.01 0.02 0.03 0.04 0.05; do
        sbatch submit.NAMD $N $A0
    done
done

