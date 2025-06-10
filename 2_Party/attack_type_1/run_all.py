import os
import sys
import subprocess
from pathlib import Path

def run_commands():
    # Get the root directory where this script is located
    root_dir = Path(__file__).parent.absolute()
    
    # Define paths relative to root directory
    shadow_model_dir = root_dir / "shadow_model_data"
    splitting_dir = root_dir / "splitting"
    
    commands = [
        {
            "file": splitting_dir / "victim_data_split.py",
            "args": ["--data-source", "victim"],
            "env": {"DATA_DIR": str(shadow_model_dir)}
        },
        {"file": root_dir / "without_dp.py", "args": []},
        {
            "file": splitting_dir / "victim_data_split.py",
            "args": ["--data-source", "shadow"],
            "env": {"DATA_DIR": str(shadow_model_dir)}
        },
        {"file": root_dir / "train_attack_model.py", "args": []},
        {
            "file": splitting_dir / "victim_data_split.py",
            "args": ["--data-source", "victim"],
            "env": {"DATA_DIR": str(shadow_model_dir)}
        },
        {"file": root_dir / "test_attack_model.py", "args": []}
    ]
    
    for cmd in commands:
        args = [sys.executable, str(cmd["file"])] + cmd["args"]
        env = os.environ.copy()
        if "env" in cmd:
            env.update(cmd["env"])
        
        print(f"\nRunning: {' '.join(args)}")
        print("=" * 50)
        
        try:
            result = subprocess.run(args, check=True, env=env)
            print(f"Completed successfully: {cmd['file'].name}")
        except subprocess.CalledProcessError as e:
            print(f"Error running {cmd['file'].name}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error running {cmd['file'].name}: {e}")
            sys.exit(1)
    
    print("\nAll commands executed successfully!")

if __name__ == "__main__":
    run_commands()
