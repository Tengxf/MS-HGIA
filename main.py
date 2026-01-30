import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import submitit
import sys
from process import processDataset
from experiments import *
from parser import get_args

os.environ["WANDB__SERVICE_WAIT"] = "300"

def main():
    # Get command line arguments
    args = get_args()
    executor = submitit.AutoExecutor(folder=args.logfolder, slurm_max_num_timeout=30)
    executor.update_parameters(
            mem_gb=args.mem,
            slurm_gpus_per_task=args.nodes,
            tasks_per_node=args.nodes,
            slurm_cpus_per_gpu=args.nodes,
            nodes=args.nodes,
            timeout_min=args.time,
            slurm_partition=args.partition,
            slurm_signal_delay_s=120,
            slurm_array_parallelism=args.job_parallel)
    executor.update_parameters(name=args.job_name)
    experiments=[]
    if args.dataset== "cam":
        experiments=experiments+launch_DASMIL_cam(args)
    elif args.dataset=='ruxian':
        experiments=experiments+launch_DASMIL_ruxian(args)
    else: 
        experiments=experiments+launch_DASMIL_lung(args)

    processDataset(experiments[0])

if __name__ == '__main__':
    main()
