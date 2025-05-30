import submitit
import os
import math

def run_batch(args_list):
    import os
    for args in args_list:
        print(f"Running: {args}")
        os.system(f"python experiments/partitioning_method/run.py {args}")

def main():
    with open("experiments/partitioning_method/experiment_args.txt") as f:
        all_args = [line.strip() for line in f if line.strip()]

    batch_size = 500
    batches = [all_args[i:i + batch_size] for i in range(0, len(all_args), batch_size)]
    print(f"Submitting {len(batches)} jobs ({batch_size} experiments per job)")

    executor = submitit.AutoExecutor(folder="submitit_logs/batched")
    executor.update_parameters(
        timeout_min=240,
        cpus_per_task=4,
        mem_gb=64,
        slurm_partition="inferno",
        slurm_account="gts-vfung3",
        name="batched-benchmark",
        slurm_array_parallelism=50
    )

    jobs = executor.map_array(run_batch, batches)
    print(f"Submitted {len(jobs)} jobs")

if __name__ == "__main__":
    main()