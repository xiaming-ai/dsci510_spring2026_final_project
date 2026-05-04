import subprocess
import os
import sys


## AI-generated with manually modification in some parts. 
def run_script(script_path):
    print(f"Running: {script_path}")

    if not os.path.exists(script_path):
        print(f"Error: Script '{script_path}' not found. Skipping")
        return False
        
    try:
        # Run the script and stream the output to the console
        result = subprocess.run([sys.executable, script_path], check=True)
        print(f"\n[SUCCESS] {script_path} completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] {script_path} failed with return code {e.returncode}.")
        return False

def main():
    # Define the pipeline order based on the README, mapping to their actual paths
    pipeline_scripts = [
        os.path.join("src", "api_access.py"),
        os.path.join("src", "clean_nhts_data.py"),
        os.path.join("src", "run.py"),
        os.path.join("src", "rf_importance.py"),  # Mentioned in README, might not exist
        os.path.join("src", "dt_importance.py"),
        "visualization.py"  # In the root directory
    ]

    print("Starting Transportation Data Pipeline...")
    
    for script in pipeline_scripts:
        if os.path.exists(script):
            success = run_script(script)
            if not success:
                print("\nPipeline stopped due to an error.")
                sys.exit(1)
        else:
            print(f"\n[WARNING] Script not found: {script}. Skipping...")
            
    print("\nPipeline execution finished!")

if __name__ == "__main__":
    main()
