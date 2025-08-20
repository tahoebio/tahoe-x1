
if [ $# -ne 2 ]; then
    echo "Usage: $0 <path_to_h5ad> <save_log_dir>"
    exit 1
fi

h5ad_path=$1 #gene_embs_v2/embs_adata.h5ad.gz
log_dir=$2

declare -a seeds=(41 42 43)  # Array of seeds
declare -a reps=("rep_1" "rep_2" "rep_3")  # Array of repetitions



for i in "${!reps[@]}"; do  # Iterate over indices of reps array
    rep=${reps[$i]}
    seed=${seeds[$i]}
    nohup python benchmark_mlp.py $h5ad_path mlp_benchmark/$log_dir/$rep --verbose --seed $seed > mlp_benchmark/$log_dir/log_$rep.txt 2>&1
done
