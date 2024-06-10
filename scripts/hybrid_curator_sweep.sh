cpu_limit=$1
graph_degree=$2
branch_factor=$3
buf_capacity=$4

sudo /home/yicheng/miniconda3/envs/ann_bench2/bin/python \
run_parallel_exp.py run_hybrid_curator_param_sweep \
    --graph_degree_space "[${graph_degree}]" \
    --branch_factor_space "[${branch_factor}]" \
    --buf_capacity_space "[${buf_capacity}]" \
    --alpha_space "[1.0]" \
    --log_dir logs/param_sweep/hybrid_curator \
    --cpu_limit ${cpu_limit} \
    --mem_limit 40000000000 \
> /dev/null 2>&1 &
