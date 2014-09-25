#!/bin/bash

# sh test.sh 4 3 2 10
max_threads=$1
max_tosses=$2
max_runs=$3
max_time=$4

file1_prefix="./res/pi_"
file2_prefix="./res/time_"

if [ $# -ne 4 ]; then
        echo "Usage: sh test.sh <max_threads> <10^max_tosses> <max_runs> <max_time>"
        exit 1
fi

# Clean data files...
data_files=$(ls ./res)
if [ ${#data_files} -gt 0 ]; then
    rm ./res/*
fi

# Perform benchmark...


for t in `seq 1 $max_threads`
do
    file1=$file1_prefix$t".csv"
    file2=$file2_prefix$t".csv"

    # Write headers.
    for i in `seq 1 $max_runs`
    do
        printf "\t, $i" >> $file1 # There's an empty (first) column...
        printf "\t, $i" >> $file2
    done
    printf "\n" >> $file1
    printf "\n" >> $file2

    for toss_order in `seq 2 $max_tosses`
    do
        printf "10^$toss_order" >> $file1
        printf "10^$toss_order" >> $file2

        fst=$t
        snd=`echo "10^$toss_order" | bc`
        f="echo $fst $snd | ./a.out"

        for j in `seq 1 $max_runs`
        do
            # Run the program with a fixed timeout...
            output=$(gtimeout $max_time bash -c "$f")

            if [ ${#output} -ne 0 ]
                then 
                    res=($output) # Convert to array (splitting at ' ')...
                    pi=${res[0]}
                    time_us=${res[1]}
                    printf ", $pi" >> $file1
                    printf ", $time_us" >> $file2
                else
                    printf ", TIMED OUT" >> $file1
                    printf ", TIMED OUT" >> $file2
                    break
            fi
        done
        printf "\n" >> $file1
        printf "\n" >> $file2
    done
done

