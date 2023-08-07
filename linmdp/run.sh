#!/bin/bash 
lambd=0.01
p=0.6
for H in 30
do
    for tau in 1
    do
        for M in 20
        do 
            echo "H = $H, tau = $tau, M = $M"
            python linmdp_main.py --H $H --tau $tau --lambd $lambd --p $p --M $M --algo LMC --start_trial 0
        done 
    done
done
