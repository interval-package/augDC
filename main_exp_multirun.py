import argparse
import subprocess
import importlib
from concurrent.futures import ThreadPoolExecutor
import os, json

def run_script(script_name, *args):
    """Run a Python script with given arguments and log output and errors to files."""
    log_file = f"{os.path.splitext(script_name)[0]}_log.txt"
    
    try:
        # Create the command to run the script with arguments
        command = ['python', script_name] + list(args)
        print(f"Running command: {' '.join(command)}")
        
        # Run the command and capture the output
        result = subprocess.run(command, capture_output=True, text=True)
        
        # Write the result and errors to the log file
        with open(log_file, 'w') as f:
            f.write(f"Output of {script_name}:\n{result.stdout}\n")
            if result.stderr:
                f.write(f"Errors of {script_name}:\n{result.stderr}\n")
                
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"Failed to run {script_name}: {e}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", default="example/tasks.json",type=str)
    # Define the scripts and their arguments
    args = parser.parse_args()
    task_file:str = args.task_file
    if task_file.endswith(".py"):
        raise NotImplementedError("py not yet.")
    elif task_file.endswith(".json"):
        with open(task_file, "rt") as f:
            config = json.load(f)
            if config["info"] == "tasks":
                tasks = config["tasks"]
            if config["info"] == "task":
                tasks = [config["task"]]
        pass
    else:
        raise NotImplementedError("This kind of file not allowed.")

    scripts_with_args = []
    for task in tasks:
        one_task = [task["script"]]
        for key, val in task["config"].items():
            one_task += [key, val]
        scripts_with_args.append(one_task)
    
    # Number of parallel threads
    num_threads = len(scripts_with_args)
    
    # Run the scripts in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(run_script, script_name, *args) for script_name, *args in scripts_with_args]
        
        # Wait for all futures to complete
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
