cpu_limit=$1
ef_construct=$2
graph_degree=$3
ef_search=$4
alpha=${5:-1.2}
filter_ef_construct=${6:-$ef_construct}

sudo /home/yicheng/miniconda3/envs/ann_bench2/bin/python \
run_parallel_exp.py run_filtered_diskann_param_sweep \
    --ef_construct_space "[${ef_construct}]" \
    --graph_degree_space "[${graph_degree}]" \
    --alpha_space "[${alpha}]" \
    --filter_ef_construct_space "[${filter_ef_construct}]" \
    --ef_search_space "[${ef_search}]" \
    --log_dir logs/param_sweep/filtered_diskann \
    --cpu_limit ${cpu_limit} \
    --mem_limit 100000000000 \
> /dev/null 2>&1 &
