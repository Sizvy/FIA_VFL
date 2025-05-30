import os
import subprocess
import shutil
from pathlib import Path

# Configuration
total_runs = 5
with_F_runs = 0
script_dir = Path(__file__).parent  

# Path configurations
shadow_data_path = script_dir / 'shadow_model_data'
victim_model_path = script_dir / 'Saved_Models'
scripts_path = script_dir / 'splitting'  
attack_script_path = script_dir / 'attack_type_4'

# Clean and create directories
shadow_data_path.mkdir(exist_ok=True)
victim_model_path.mkdir(exist_ok=True)

def run_process(command, cwd=None):
    process = subprocess.Popen(command, shell=True, cwd=cwd)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")

for run_id in range(1, total_runs + 1):
    print(f"\n=== Starting Run {run_id}/{total_runs} ===")
    
    include_F = run_id <= with_F_runs
    
    print("Running shadow_data_split.py...")
    if include_F:
        run_process(f"python3 shadow_data_split_exp.py --keep_target_feature", cwd=scripts_path)
    else:
        run_process(f"python3 shadow_data_split_exp.py", cwd=scripts_path)
    
    print("Running victim_data_split.py...")
    run_process("python3 victim_data_split.py", cwd=scripts_path)
    
    print("Training victim model...")
    run_process("python3 without_dp.py")
    
    print("Testing Attack ...")
    run_process("python3 step_4.py", cwd=attack_script_path)
    # Rename model file
    # src = victim_model_path / "best_vfl_model.pt"
    # dst = victim_model_path / f"best_vfl_model_{run_id}.pt"
    # shutil.move(str(src), str(dst))
    
    # print(f"Completed run {run_id}. Model saved as {dst}")

print("\nExperiment complete!")
print(f"- 1-{with_F_runs}: With target feature")
print(f"- {with_F_runs+1}-{total_runs}: Without target feature")
