#!/bin/sh

#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=4

MASTER=`/bin/hostname -s`
MPORT=$(shuf -i 6000-9999 -n 1)

jobdir="$(dirname "$(dirname "$(pwd)")")";
log_path="/scratch/pritam/OUTPUTS/logs"

CONFIG=$1

echo "
cd $HOME; cd $jobdir;
python main_xkd.py \
            --dist-url tcp://${MASTER}:${MPORT} \
            --dist-backend 'nccl' \
            --multiprocessing-distributed \
            --world-size ${SLURM_JOB_NUM_NODES} --rank \${SLURM_PROCID} \
            --job_id ${SLURM_ARRAY_JOB_ID} \
            --quiet --sub_dir 'pretext' \
            --config-file ${CONFIG} \
            --seed 99999 \
            --db 'kinetics400' |& tee ${log_path}/${SLURM_ARRAY_JOB_ID}_\${SLURM_PROCID}.out
            
if (( ${SLURM_PROCID} != 0 ));then         
    while [ ! -f "${log_path}/${SLURM_JOBID}_0.out" ]
        do
        sleep 2
        echo 'waiting for rank 0'
    done
fi
            
            " > srun_worker_${SLURM_ARRAY_JOB_ID}_${SLURM_PROCID}.sh
srun bash srun_worker_${SLURM_ARRAY_JOB_ID}_${SLURM_PROCID}.sh
