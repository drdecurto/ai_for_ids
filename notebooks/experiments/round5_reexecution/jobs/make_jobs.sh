# Generates the production Slurm scripts on the cluster under $S/logs/jobs and a submit helper.
# Usage (on the login node): bash make_jobs.sh
S=/path/to/scratch/ids; J=$S/logs/jobs; mkdir -p $J $S/runs/r5
CELLS="cic_ids2017_V1 cic_ids2017_V3 cic_unsw_V1 cic_unsw_V2"
CLASSICAL=rf_pruned_full,rf_strong_full,xgb_full,rf_pruned_balanced,rf_strong_balanced,xgb_balanced

# ---- classical panel: one CPU job per split; the four cells run as parallel processes (28 cores each)
for split in row2026 temporal row2027 row2028 hostpair; do
CL="$CELLS"; [ $split = hostpair ] && CL="cic_unsw_V1 cic_unsw_V2"
cat > $J/cls_$split.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=ids-cls-$split
#SBATCH --account=<account>
#SBATCH --qos=gp_debug
#SBATCH --partition=gpp
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=28
#SBATCH --time=02:00:00
#SBATCH --output=$S/logs/%x-%j.out
#SBATCH --error=$S/logs/%x-%j.err
set -uo pipefail
source $S/env.sh; export OMP_NUM_THREADS=28
cd $S/code/ids_round5
echo "start \$(date) node \$(hostname)"
for cell in $CL; do
  srun --exact --ntasks=1 --cpus-per-task=28 --cpu-bind=none python run_panel.py --cell \$cell --split $split \\
     --models $CLASSICAL --seeds 0,1,2,3,4,5,6,7,8,9 --out $S/runs/r5 --full-validate-seeds 0,1,2 > $S/logs/cls_${split}_\$cell.log 2>&1 &
done
wait
echo "end \$(date)"
SB
done

# ---- foundation models: one GPU job per (split, seed block); 4 GPUs = 4 cells in parallel; resumable
for split in row2026 temporal row2027 row2028 hostpair; do
CL="$CELLS"; [ $split = hostpair ] && CL="cic_unsw_V1 cic_unsw_V2"
for blk in 0-4 5-9; do
seeds=$(echo $blk | awk -F- '{for(i=$1;i<=$2;i++){printf "%s%d",(i>$1?",":""),i}}')
cat > $J/fm_${split}_s${blk}.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=ids-fm-$split-$blk
#SBATCH --account=<account>
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=$S/logs/%x-%j.out
#SBATCH --error=$S/logs/%x-%j.err
set -uo pipefail
source $S/env.sh; export HF_HUB_OFFLINE=1
cd $S/code/ids_round5
echo "start \$(date) node \$(hostname)"
k=0
for cell in $CL; do
  CUDA_VISIBLE_DEVICES=\$k srun --exact --ntasks=1 --gres=gpu:1 --cpus-per-task=20 --cpu-bind=none python run_panel.py --cell \$cell --split $split \\
     --models tabpfn_2.5,tabicl --seeds $seeds --out $S/runs/r5 > $S/logs/fm_${split}_${blk}_\$cell.log 2>&1 &
  k=\$((k+1))
done
wait
echo "end \$(date)"
SB
done; done

# ---- leakage audit (GPU NN search): all cells x {row2026, temporal}, seed 0, one job
cat > $J/leak.sbatch <<SB
#!/usr/bin/env bash
#SBATCH --job-name=ids-leak
#SBATCH --account=<account>
#SBATCH --qos=acc_debug
#SBATCH --partition=acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=$S/logs/%x-%j.out
#SBATCH --error=$S/logs/%x-%j.err
set -uo pipefail
source $S/env.sh
cd $S/code/ids_round5
for cell in $CELLS; do for split in row2026 temporal; do
  srun --ntasks=1 --cpu-bind=none python leakage_audit.py --cell \$cell --split \$split --seed 0 --out $S/audit
done; done
for cell in cic_unsw_V1 cic_unsw_V2; do srun --ntasks=1 --cpu-bind=none python leakage_audit.py --cell \$cell --split hostpair --seed 0 --out $S/audit; done
SB

# ---- queue keeper: submits the acc_debug jobs one after another (acc_debug admits one job per user)
cat > $J/keeper.sh <<'KP'
#!/usr/bin/env bash
# bash keeper.sh job1.sbatch job2.sbatch ...   (run detached on the login node)
S=/path/to/scratch/ids
for j in "$@"; do
  while squeue -u $USER -h -q acc_debug | grep -q .; do sleep 120; done
  jid=$(sbatch --parsable $j); echo "$(date) submitted $j -> $jid" >> $S/logs/keeper.log
  sleep 60
  while squeue -u $USER -h -j $jid | grep -q .; do sleep 120; done
  echo "$(date) finished $j ($jid)" >> $S/logs/keeper.log
done
echo "$(date) keeper done" >> $S/logs/keeper.log
KP
ls $J
