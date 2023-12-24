```
python main_xkd.py \
            --dist-url tcp://${MASTER}:${MPORT} \
            --dist-backend 'nccl' \
            --multiprocessing-distributed \
            --world-size ${SLURM_JOB_NUM_NODES} --rank \${SLURM_PROCID} \
            --job_id ${SLURM_ARRAY_JOB_ID} \
            --quiet --sub_dir 'pretext' \
            --config-file 'xkd.yaml' \
            --seed 99999 \
            --db 'kinetics400' 
                
```
