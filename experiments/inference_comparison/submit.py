import submitit
import os
import math

def run_batch(args_list):
    import os
    for args in args_list:
        print(f"Running: {args}")
        os.system(f"python experiments/inference_comparison/run.py {args}")

def main():
    with open("experiments/inference_comparison/experiment_args.txt") as f:
        all_args = [line.strip() for line in f if line.strip()]

    batch_size = 500
    batches = [all_args[i:i + batch_size] for i in range(0, len(all_args), batch_size)]
    print(f"Submitting {len(batches)} jobs ({batch_size} experiments per job)")

    executor = submitit.AutoExecutor(folder="experiments/inference_comparison/logs/batched")
    executor.update_parameters(
        timeout_min=240,
        cpus_per_task=2,
        slurm_account="gts-vfung3",
        name="inference-comparison",
        slurm_array_parallelism=50,
        
        gres='gpu:H200:1',
        
        slurm_additional_parameters={
            "mem-per-gpu": "200G"
        }
    )

    jobs = executor.map_array(run_batch, batches)
    print(f"Submitted {len(jobs)} jobs")

if __name__ == "__main__":
    main()