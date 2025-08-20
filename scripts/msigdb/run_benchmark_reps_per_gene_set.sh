declare -a seeds=(41 42 43 44)  # Array of seeds
declare -a reps=("rep_1" "rep_2" "rep_3" "rep_4")  # Array of repetitions
declare -a sets=("BP" "CC" "MF")  # Array of sets
declare -a seeds=(44)  # Array of seeds
declare -a reps=("rep_4")  # Array of repetitions
declare -a sets=("BP" "CC" "MF")  # Array of sets
for i in "${!reps[@]}"; do  # Iterate over indices of reps array
    rep=${reps[$i]}
    seed=${seeds[$i]}
    for j in "${!sets[@]}"; do  # Iterate over indices of reps array
        set=${sets[$j]}
        nohup python benchmark_mlp.py gene_embs_v2/c5/$set/embs_adata.h5ad.gz mlp_benchmark/v6/c5/$set/$rep --verbose --seed $seed > mlp_benchmark/v6/c5/$set/log_$rep.txt 2>&1
    done
done