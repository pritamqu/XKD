#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1

export MASTER_ADDR=$(hostname)
echo "rank $SLURM_NODEID master: $MASTER_ADDR"
echo "rank $SLURM_NODEID Launching python script"

MASTER=`/bin/hostname -s`
MPORT=$(shuf -i 6000-9999 -n 1)
jobdir="$(dirname "$(dirname "$(pwd)")")";
log_path="./OUTPUTS/logs"

cd $HOME
cd $jobdir

CONFIG=$1
WEIGHT_PATH="/path.pth.tar"

python eval_svm_video.py \
--world-size 1 --rank 0 --gpu 0 \
--job_id ${SLURM_JOBID} --quiet --sub_dir 'svm' \
--db 'hmdb51' \
--config-file ${CONFIG} \
--weight_path ${WEIGHT_PATH} \
--seed 99999
