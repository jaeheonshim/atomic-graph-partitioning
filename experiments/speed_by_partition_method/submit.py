import submitit
import os

def run_experiment(args_line: str):
    print(f"Running: {args_line}")
    os.system(f"python benchmark_materials.py {args_line}")

def main():
    with open("experiment_args.txt") as f:
        all_args = [line.strip() for line in f if line.strip()]

    executor = submitit.AutoExecutor(folder="submitit_logs")
    executor.update_parameters(
        timeout_min=480,
        cpus_per_task=4,
        mem_gb=64,
        slurm_partition="inferno",
        slurm_account="gts-vfung3",
        name="partition-benchmark",
    )

    jobs = executor.map_array(run_experiment, all_args)
    print(f"Submitted {len(jobs)} jobs")

if __name__ == "__main__":
    main()