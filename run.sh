#!/bin/bash
# Source the conda.sh script to enable 'conda' command
source /net/csefiles/coc-fung-cluster/lingyu/miniconda3/etc/profile.d/conda.sh


desired_partition_list=(
    20
    30
    40
    50
)

num_mp_list=(
    3
)

expected_unit_size_list=(
    100000
)


conda activate mattersim-tune
for desired_partition in "${desired_partition_list[@]}"
do
    for num_mp in "${num_mp_list[@]}"
    do
        for expected_unit_size in "${expected_unit_size_list[@]}"
        do
            python mattersim_lingyu.py \
            --desired_partitions $desired_partition \
            --num_message_passing $num_mp \
            --expected_unit_size $expected_unit_size
        done
    done
done