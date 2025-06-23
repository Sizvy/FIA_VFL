import os
import subprocess
import shutil
from pathlib import Path

script_dir = Path(__file__).parent

scripts_path = script_dir / 'splitting'  
attack_script_path = script_dir / 'attack_type_4'
shadow_data_path = script_dir / 'shadow_model_data'

# Clean and create directories
shadow_data_path.mkdir(exist_ok=True)

def run_process(command, cwd=None):
    process = subprocess.Popen(command, shell=True, cwd=cwd)
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")
    
print("Running shadow_data_split.py...")
run_process(f"python3 shadow_data_split.py", cwd=scripts_path)
print("Running Step 1 ...")
run_process("python3 step_1_extra.py", cwd=attack_script_path)
run_process("python3 step_1.py", cwd=attack_script_path)
print("Running Step 2 ...")
run_process("python3 step_2_extra.py", cwd=attack_script_path)
print("Running Step 3 ...")
run_process("python3 step_3_extra.py", cwd=attack_script_path)
print("\n Training completed!")
