rank=$1
cpu_mask=$2
ivf_cluster_size=$3
ivf_max_iter=$4
graph_degree=$5

taskset $cpu_mask \
python -m benchmark.profile_parlay_ivf \
    exp_parlay_ivf_param_sweep_worker \
        --rank $rank \
        --ivf_cluster_size $ivf_cluster_size \
        --ivf_max_iter $ivf_max_iter \
        --graph_degree $graph_degree \
        --output_path "parlay_ivf_param_sweep.csv" &
