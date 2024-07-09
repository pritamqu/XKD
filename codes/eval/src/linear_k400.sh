#!/bin/sh

#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=4


export MASTER_ADDR=$(hostname)
echo "rank $SLURM_NODEID master: $MASTER_ADDR"
echo "rank $SLURM_NODEID Launching python script"

MASTER=`/bin/hostname -s`
MPORT=$(shuf -i 6000-9999 -n 1)
jobdir="$(dirname "$(dirname "$(pwd)")")";
log_path="./OUTPUTS/logs"

CONFIG=$1
WEIGHT_PATH="/path.pth.tar"

echo "

cd $HOME; cd $jobdir;
python eval_linear_video.py \
            --dist-url tcp://${MASTER}:${MPORT} \
            --dist-backend 'nccl' \
            --multiprocessing-distributed \
            --world-size ${SLURM_JOB_NUM_NODES} --rank \${SLURM_PROCID} \
            --job_id ${SLURM_JOBID} \
            --quiet --sub_dir 'linear' \
            --config-file ${CONFIG} \
            --weight_path ${WEIGHT_PATH} \
            --db 'kinetics400' |& tee ${log_path}/${SLURM_JOBID}_\${SLURM_PROCID}.out

if (( ${SLURM_PROCID} != 0 ));then         
    while [ ! -f "${log_path}/${SLURM_JOBID}_0.out" ]
        do
        sleep 2
        echo 'waiting for rank 0'
    done
fi
            " > srun_worker_${SLURM_JOBID}_${SLURM_PROCID}.sh
srun --mem 167G bash srun_worker_${SLURM_JOBID}_${SLURM_PROCID}.sh

